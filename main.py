from flask import Flask, render_template, request, redirect, url_for, send_file
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import ezdxf
import os
from scipy.spatial import cKDTree
import numpy as np

app = Flask(__name__)

# 地區與CSV文件的映射
REGION_CSV_MAP = {
    '台中': 'TC.csv',
    '台南': 'TN.csv',
    '高雄': 'KH.csv',
    '雲林': 'YL.csv'
}

@app.route('/')
def index():
    # 傳遞地區選項到前端
    return render_template('index.html', regions=list(REGION_CSV_MAP.keys()))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    region = request.form.get('region')  # 獲取選擇的地區

    if file.filename == '':
        return redirect(url_for('index'))

    if file and file.filename.endswith('.dxf'):
        # 根據地區選擇對應的CSV文件
        csv_file_path = os.path.join('data', REGION_CSV_MAP.get(region, 'default.csv'))
        wtc_csv_file_path = os.path.join('data', REGION_CSV_MAP.get(region, 'default.csv').replace('.csv', '_WTC.csv'))

        # 檢查已接管CSV檔案是否存在，如果不存在則使用預設路徑
        if not os.path.exists(wtc_csv_file_path):
            wtc_csv_file_path = os.path.join('data', 'default_WTC.csv')

        # 將上傳的DXF檔保存到伺服器
        dxf_file_path = os.path.join('uploads', file.filename)
        file.save(dxf_file_path)

        # 執行分析和檔生成 (原始門牌資料)
        intersection, grouped = perform_common_analysis(dxf_file_path, csv_file_path)
        result = analyze_dxf_result(intersection, grouped)

        # 執行分析和檔生成 (已接管門牌資料)
        wtc_intersection, wtc_grouped = perform_common_analysis(dxf_file_path, wtc_csv_file_path)
        wtc_result = analyze_dxf_result(wtc_intersection, wtc_grouped)

        # 檢查是否兩組結果皆為空（所有值為 0）
        warning_message = None
        if isinstance(result, dict) and all(v == 0 for v in result.values()) and \
           isinstance(wtc_result, dict) and all(v == 0 for v in wtc_result.values()):
            warning_message = "空間交集為空集合，沒有門牌點在DXF範圍內。請檢查DXF範圍線是否正確，或確認所選縣市是否正確。"

        # 檢查分析結果是否為空，如果是空結果，設定為 None
        if not isinstance(result, dict):
            result = None
        if not isinstance(wtc_result, dict):
            wtc_result = None

        new_dxf_path = analyze_and_export_dxf(dxf_file_path, csv_file_path, intersection, grouped)
        
        # 刪除臨時上傳文件
        os.remove(dxf_file_path)

        # 更新鄉鎮市區統計的已接管戶數和接管率
        if result and wtc_result and 'district_stats' in result:
            district_stats = result['district_stats']
            for district in district_stats:
                district_code = district['code']
                # 計算已接管戶數
                if not wtc_intersection.empty and '鄉鎮市區代碼' in wtc_intersection.columns:
                    wtc_district_data = wtc_intersection[wtc_intersection['鄉鎮市區代碼'] == district_code]
                    wtc_district_grouped = wtc_district_data.groupby('group').size().reset_index(name='count')
                    district['wtc_houses'] = int(wtc_district_grouped['count'].sum()) if not wtc_district_grouped.empty else 0
                # 計算接管率
                if district['total_houses'] > 0:
                    district['connection_rate'] = round((district['wtc_houses'] / district['total_houses']) * 100, 3)
                else:
                    district['connection_rate'] = 0.000

        # 渲染結果頁面，並提供生成的 DXF 檔下載鏈接，同時傳遞分析結果和警告訊息
        return render_template('result.html',
                              result=result,
                              wtc_result=wtc_result,
                              download_link=os.path.basename(new_dxf_path),
                              warning_message=warning_message,
                              district_stats=result['district_stats'] if result and 'district_stats' in result else None)
    else:
        return redirect(url_for('index'))

def perform_common_analysis(dxf_file_path, csv_file_path):
    # 讀取DXF檔案並轉換為多邊形範圍 
    dxf_doc = ezdxf.readfile(dxf_file_path)
    msp = dxf_doc.modelspace()
    polygons = []
    for entity in msp.query("LWPOLYLINE"):
        points = [(p[0], p[1]) for p in entity]
        polygons.append(Polygon(points))

    dxf_gdf = gpd.GeoDataFrame(geometry=polygons)

    # 優化 CSV 檔案讀取: 根據檔案內容決定要讀取的欄位
    required_columns = ['X', 'Y']
    csv_data = pd.read_csv(csv_file_path)
    
    # 檢查CSV是否包含鄉鎮市區代碼欄位
    if '鄉鎮市區代碼' in csv_data.columns:
        required_columns.append('鄉鎮市區代碼')
        csv_data = csv_data[required_columns]
        csv_data = csv_data.astype({'X': float, 'Y': float, '鄉鎮市區代碼': str})
    else:
        csv_data = csv_data[required_columns]
        csv_data = csv_data.astype({'X': float, 'Y': float})
        
    csv_data['geometry'] = gpd.points_from_xy(csv_data['X'], csv_data['Y'])
    gdf_csv = gpd.GeoDataFrame(csv_data, geometry='geometry')

    # 進行空間交集分析 
    intersection = gpd.sjoin(gdf_csv, dxf_gdf, how='inner', predicate='intersects')

    # 提取座標點 
    coords = np.array([(geom.x, geom.y) for geom in intersection.geometry])

    # 檢查 coords 陣列是否為空（移除 print 警告）
    if coords.size == 0:
        return gpd.GeoDataFrame(), pd.DataFrame()

    # 使用KDTree找出0.17m內的鄰居都視為同戶 
    tree = cKDTree(coords)
    groups_indices = tree.query_ball_tree(tree, r=0.17)

    # 優化組ID分配
    group_ids = np.zeros(len(intersection), dtype=int) - 1
    group_counter = 0
    for i, neighbors in enumerate(groups_indices):
        if group_ids[i] == -1:
            group_ids[i] = group_counter
            group_ids[neighbors] = group_counter
            group_counter += 1

    intersection['group'] = group_ids

    # 按新的組進行分組，計算每組的門牌數量 
    grouped = intersection.groupby('group').size().reset_index(name='count')

    return intersection, grouped

def analyze_dxf_result(intersection, grouped):
    # 檢查 grouped 是否為空 DataFrame（移除 print 警告）
    if grouped.empty:
        return {
            'num_houses': 0,
            'num_house_buildings': 0,
            'num_apartments': 0,
            'num_apartment_buildings': 0,
            'num_buildings': 0,
            'num_building_structures': 0,
            'total_houses': 0,
            'total_buildings': 0,
            'district_stats': []
        }

    # 計算透天、公寓和大樓的戶數與棟數 
    num_houses = int(grouped[grouped['count'] == 1]['count'].sum())
    num_house_buildings = int(grouped[grouped['count'] == 1].shape[0])

    num_apartments = int(grouped[(grouped['count'] >= 2) & (grouped['count'] <= 6)]['count'].sum())
    num_apartment_buildings = int(grouped[(grouped['count'] >= 2) & (grouped['count'] <= 6)].shape[0])

    num_buildings = int(grouped[grouped['count'] >= 7]['count'].sum())
    num_building_structures = int(grouped[grouped['count'] >= 7].shape[0])

    # 計算總計 
    total_houses = num_houses + num_apartments + num_buildings
    total_buildings = num_house_buildings + num_apartment_buildings + num_building_structures

    # 計算鄉鎮市區統計
    district_stats = []
    if not intersection.empty:
        if '鄉鎮市區代碼' in intersection.columns:
            district_groups = intersection.groupby('鄉鎮市區代碼')
            for district_code, district_data in district_groups:
                district_grouped = district_data.groupby('group').size().reset_index(name='count')
                district_total = int(district_grouped['count'].sum())
                district_stats.append({
                    'code': district_code,
                    'total_houses': district_total,
                    'wtc_houses': 0,  # 這個值會在後續處理中更新
                    'connection_rate': 0.0  # 這個值會在後續處理中更新
                })
        else:
            # 如果沒有鄉鎮市區代碼，將所有資料視為一個區域
            total_grouped = intersection.groupby('group').size().reset_index(name='count')
            total_houses = int(total_grouped['count'].sum())
            district_stats.append({
                'code': 'ALL',
                'total_houses': total_houses,
                'wtc_houses': 0,
                'connection_rate': 0.0
            })

    # 返回結果 
    return {
        'num_houses': num_houses,
        'num_house_buildings': num_house_buildings,
        'num_apartments': num_apartments,
        'num_apartment_buildings': num_apartment_buildings,
        'num_buildings': num_buildings,
        'num_building_structures': num_building_structures,
        'total_houses': total_houses,
        'total_buildings': total_buildings,
        'district_stats': district_stats
    }

def analyze_and_export_dxf(dxf_file_path, csv_file_path, intersection, grouped):
    # 讀取DXF文件 
    dxf_doc = ezdxf.readfile(dxf_file_path)
    msp = dxf_doc.modelspace()

    # 在原 DXF 文件中繪製交集的門牌座標點，並根據數量設置顏色 
    for idx, row in intersection.iterrows():
        point = row['geometry']
        group_id = row['group']
        count = grouped[grouped['group'] == group_id]['count'].values[0]
        if count == 1:
            color = 7  # 白色
        elif 2 <= count <= 6:
            color = 30  # 橘色
        else:
            color = 3  # 綠色
        msp.add_circle((point.x, point.y), radius=0.5, dxfattribs={'color': color})

    # 定義新生成的 DXF 檔路徑 
    new_dxf_path = os.path.join('outputs', 'output_with_points.dxf')

    # 保存新的 DXF 檔 
    dxf_doc.saveas(new_dxf_path)

    return new_dxf_path

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('outputs', filename), as_attachment=True)

if __name__ == '__main__':
    # 確保必要的目錄存在 
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # 本地測試使用 Flask 內置服務器 
    app.run(host='0.0.0.0', port=5000, debug=True)
    # 部署到雲端時，註釋掉上面這行，使用 Gunicorn 運行：
    # gunicorn --bind 0.0.0.0:5000 main:app