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
        wtc_csv_file_path = os.path.join('data', REGION_CSV_MAP.get(region, 'default.csv').replace('.csv', '_WTC.csv')) # 假設已接管CSV檔名為 原檔名_WTC.csv

        # 檢查已接管CSV檔案是否存在，如果不存在則使用預設路徑或給予提示
        if not os.path.exists(wtc_csv_file_path):
            wtc_csv_file_path = os.path.join('data', 'default_WTC.csv') # 或是您可以設定一個預設的已接管CSV路徑，或者提示使用者

        # 將上傳的DXF檔保存到伺服器
        dxf_file_path = os.path.join('uploads', file.filename) # 修改變數名稱 file_path -> dxf_file_path 更明確
        file.save(dxf_file_path)

        # 執行分析和檔生成 (原始門牌資料)
        intersection, grouped = perform_common_analysis(dxf_file_path, csv_file_path)
        result = analyze_dxf_result(intersection, grouped)  # 進行門牌分析

        # 執行分析和檔生成 (已接管門牌資料)
        wtc_intersection, wtc_grouped = perform_common_analysis(dxf_file_path, wtc_csv_file_path) # 使用不同的CSV路徑
        wtc_result = analyze_dxf_result(wtc_intersection, wtc_grouped)  # 進行已接管門牌分析

        # 檢查分析結果是否為空，如果是空結果，可以設定為 None 或其他預設值
        if not isinstance(result, dict): # 判斷 result 是否為字典，如果不是字典，就表示是空結果
            result = None # 或者設定為其他預設值，例如空字典 {}
        if not isinstance(wtc_result, dict): # 判斷 wtc_result 是否為字典，如果不是字典，就表示是空結果
            wtc_result = None # 或者設定為其他預設值，例如空字典 {}

        new_dxf_path = analyze_and_export_dxf(dxf_file_path, csv_file_path, intersection, grouped)  # 生成包含座標點的DXF檔 (這邊仍然使用原始門牌資料的intersection和grouped來標註顏色，您可以根據需求調整)

        # 刪除臨時上傳文件
        os.remove(dxf_file_path)

        # 渲染結果頁面，並提供生成的 DXF 檔下載鏈接，同時傳遞兩組分析結果
        return render_template('result.html',
                               result=result, # 原始門牌分析結果
                               wtc_result=wtc_result, # 已接管門牌分析結果
                               download_link=os.path.basename(new_dxf_path))
    else:
        return redirect(url_for('index'))

def perform_common_analysis(dxf_file_path, csv_file_path):
    # 讀取DXF檔案並轉換為多邊形範圍 (這部分程式碼與之前相同，沒有修改)
    dxf_doc = ezdxf.readfile(dxf_file_path)
    msp = dxf_doc.modelspace()
    polygons = []
    for entity in msp.query("LWPOLYLINE"):
        points = [(p[0], p[1]) for p in entity]
        polygons.append(Polygon(points))

    dxf_gdf = gpd.GeoDataFrame(geometry=polygons)

    # 優化 CSV 檔案讀取: 加入 dtype 參數 (這部分程式碼與之前相同，沒有修改)
    csv_data = pd.read_csv(
        csv_file_path,
        usecols=['X', 'Y'],  # 僅讀取 X, Y 欄位
        dtype={'X': float, 'Y': float}  # 顯式指定 X, Y 欄位為 float 類型
    )
    csv_data['geometry'] = gpd.points_from_xy(csv_data['X'], csv_data['Y'])
    gdf_csv = gpd.GeoDataFrame(csv_data, geometry='geometry')

    # 進行空間交集分析 (這部分程式碼與之前相同，沒有修改)
    intersection = gpd.sjoin(gdf_csv, dxf_gdf, how='inner', predicate='intersects')

    # 提取座標點 (這部分程式碼與之前相同，沒有修改)
    coords = np.array([(geom.x, geom.y) for geom in intersection.geometry])

    # **檢查 coords 陣列是否為空**
    if coords.size == 0:
        print("警告：空間交集分析結果為空，沒有門牌點在DXF範圍內。") # 可選的警告訊息
        return gpd.GeoDataFrame(), pd.DataFrame() # 返回空的 GeoDataFrame 和 DataFrame

    # 使用KDTree找出0.17m內的鄰居都視為同戶 (這部分程式碼與之前相同，沒有修改)
    tree = cKDTree(coords)
    groups_indices = tree.query_ball_tree(tree, r=0.17)

    # 優化組ID分配: 更簡潔的版本 (組ID可能不連續) (這部分程式碼與之前相同，沒有修改)
    group_ids = np.zeros(len(intersection), dtype=int) - 1  # 初始化為 -1
    group_counter = 0
    for i, neighbors in enumerate(groups_indices):
        if group_ids[i] == -1:  # 如果點 i 還沒有被分配組
            group_ids[i] = group_counter
            group_ids[neighbors] = group_counter  # 將鄰居點分配相同組ID
            group_counter += 1

    intersection['group'] = group_ids

    # 按新的組進行分組，計算每組的門牌數量 (這部分程式碼與之前相同，沒有修改)
    grouped = intersection.groupby('group').size().reset_index(name='count')

    return intersection, grouped

def analyze_dxf_result(intersection, grouped):
    # **檢查 grouped 是否為空 DataFrame**
    if grouped.empty:
        print("警告：grouped DataFrame 為空，無法進行門牌分析。") # 可選的警告訊息
        return { # 返回預設的空結果
            'num_houses': 0,
            'num_house_buildings': 0,
            'num_apartments': 0,
            'num_apartment_buildings': 0,
            'num_buildings': 0,
            'num_building_structures': 0,
            'total_houses': 0,
            'total_buildings': 0
        }

    # 計算透天、公寓和大樓的戶數與棟數 (這部分程式碼與之前相同，沒有修改)
    num_houses = int(grouped[grouped['count'] == 1]['count'].sum())  # 透天的戶數
    num_house_buildings = int(grouped[grouped['count'] == 1].shape[0])  # 透天的棟數

    num_apartments = int(grouped[(grouped['count'] >= 2) & (grouped['count'] <= 6)]['count'].sum())  # 公寓的戶數
    num_apartment_buildings = int(grouped[(grouped['count'] >= 2) & (grouped['count'] <= 6)].shape[0])  # 公寓的棟數

    num_buildings = int(grouped[grouped['count'] >= 7]['count'].sum())  # 大樓的戶數
    num_building_structures = int(grouped[grouped['count'] >= 7].shape[0])  # 大樓的棟數

    # 計算總計 (這部分程式碼與之前相同，沒有修改)
    total_houses = num_houses + num_apartments + num_buildings
    total_buildings = num_house_buildings + num_apartment_buildings + num_building_structures

    # 返回結果 (這部分程式碼與之前相同，沒有修改)
    return {
        'num_houses': num_houses,
        'num_house_buildings': num_house_buildings,
        'num_apartments': num_apartments,
        'num_apartment_buildings': num_apartment_buildings,
        'num_buildings': num_buildings,
        'num_building_structures': num_building_structures,
        'total_houses': total_houses,
        'total_buildings': total_buildings
    }

def analyze_and_export_dxf(dxf_file_path, csv_file_path, intersection, grouped):
    # 讀取DXF文件 (這部分程式碼與之前相同，沒有修改)
    dxf_doc = ezdxf.readfile(dxf_file_path)
    msp = dxf_doc.modelspace()

    # 在原 DXF 文件中繪製交集的門牌座標點，並根據數量設置顏色 (這部分程式碼與之前相同，沒有修改)
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

    # 定義新生成的 DXF 檔路徑 (這部分程式碼與之前相同，沒有修改)
    new_dxf_path = os.path.join('outputs', 'output_with_points.dxf')

    # 保存新的 DXF 檔 (這部分程式碼與之前相同，沒有修改)
    dxf_doc.saveas(new_dxf_path)

    return new_dxf_path

@app.route('/download/<filename>')
def download_file(filename):
    # 確保只拼接一次 'outputs' (這部分程式碼與之前相同，沒有修改)
    return send_file(os.path.join('outputs', filename), as_attachment=True)

if __name__ == '__main__':
    # 確保必要的目錄存在 (這部分程式碼與之前相同，沒有修改)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # 本地測試使用 Flask 內置服務器 (這部分程式碼與之前相同，沒有修改)
    app.run(host='0.0.0.0', port=5000, debug=True)
    # 部署到雲端時，註釋掉上面這行，使用 Gunicorn 運行：
    # gunicorn --bind 0.0.0.0:5000 main:app