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
    '雲林': 'YI_202311.csv'
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

        # 將上傳的DXF檔保存到伺服器
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # 執行分析和檔生成
        intersection, grouped = perform_common_analysis(file_path, csv_file_path)
        result = analyze_dxf_result(intersection, grouped)  # 進行門牌分析
        new_dxf_path = analyze_and_export_dxf(file_path, csv_file_path, intersection, grouped)  # 生成包含座標點的DXF檔

        # 刪除臨時上傳文件
        os.remove(file_path)

        # 渲染結果頁面，並提供生成的 DXF 檔下載鏈接
        return render_template('result.html', result=result, download_link=os.path.basename(new_dxf_path))
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

    # 優化 CSV 檔案讀取: 加入 dtype 參數
    csv_data = pd.read_csv(
        csv_file_path,
        usecols=['X', 'Y'],  # 僅讀取 X, Y 欄位
        dtype={'X': float, 'Y': float}  # 顯式指定 X, Y 欄位為 float 類型
    )
    csv_data['geometry'] = gpd.points_from_xy(csv_data['X'], csv_data['Y'])
    gdf_csv = gpd.GeoDataFrame(csv_data, geometry='geometry')


    # 進行空間交集分析
    intersection = gpd.sjoin(gdf_csv, dxf_gdf, how='inner', predicate='intersects')

    # 提取座標點
    coords = np.array([(geom.x, geom.y) for geom in intersection.geometry])

    # 使用KDTree找出0.17m內的鄰居都視為同戶
    tree = cKDTree(coords)
    groups_indices = tree.query_ball_tree(tree, r=0.17)

    # 優化組ID分配: 更簡潔的版本 (組ID可能不連續)
    group_ids = np.zeros(len(intersection), dtype=int) - 1 # 初始化為 -1
    group_counter = 0
    for i, neighbors in enumerate(groups_indices):
        if group_ids[i] == -1: # 如果點 i 還沒有被分配組
            group_ids[i] = group_counter
            group_ids[neighbors] = group_counter # 將鄰居點分配相同組ID
            group_counter += 1

    intersection['group'] = group_ids


    # 按新的組進行分組，計算每組的門牌數量
    grouped = intersection.groupby('group').size().reset_index(name='count')

    return intersection, grouped


def analyze_dxf_result(intersection, grouped):
    # 計算透天、公寓和大樓的戶數與棟數
    num_houses = int(grouped[grouped['count'] == 1]['count'].sum())  # 透天的戶數
    num_house_buildings = int(grouped[grouped['count'] == 1].shape[0])  # 透天的棟數

    num_apartments = int(grouped[(grouped['count'] >= 2) & (grouped['count'] <= 6)]['count'].sum())  # 公寓的戶數
    num_apartment_buildings = int(grouped[(grouped['count'] >= 2) & (grouped['count'] <= 6)].shape[0])  # 公寓的棟數

    num_buildings = int(grouped[grouped['count'] >= 7]['count'].sum())  # 大樓的戶數
    num_building_structures = int(grouped[grouped['count'] >= 7].shape[0])  # 大樓的棟數

    # 計算總計
    total_houses = num_houses + num_apartments + num_buildings
    total_buildings = num_house_buildings + num_apartment_buildings + num_building_structures

    # 返回結果
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
    # 確保只拼接一次 'outputs'
    return send_file(os.path.join('outputs', filename), as_attachment=True)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    app.run(host='0.0.0.0', port=5000, debug=True)
