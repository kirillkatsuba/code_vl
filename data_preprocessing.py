import numpy as np
import os
import json

def read_pts_file(file_path):
    """
    Read pts, return tuple = (arrray of x and y coords;  number of points for the image)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    points =[]
    is_data = False
    n_points = 0
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('n_points:'):
            n_points = int(line.split(':')[1].strip())
            
        elif line == '{':
            is_data = True
            
        elif line == '}':
            is_data = False
            
        elif is_data:
            parts = line.split()
            if len(parts) == 2:
                x = float(parts[0])
                y = float(parts[1])
                points.append([x, y])
                
    points_array = np.array(points, dtype=np.float32)
    
    assert len(points_array) == n_points, f"Ошибка в файле {file_path}: ожидалось {n_points}, найдено {len(points_array)}"
    
    return points_array, n_points


if __name__ == '__main__': 
    dataset_names = {
        'Menpo': {
            'test': [],
            'train': []
        },
        '300W': {
            'test': [],
            'train': []
        }
    }

    data_names = dataset_names.keys()
    for name in data_names:
        for name_process in ['train', 'test']:
            images = os.listdir(f'{name}/{name_process}')
            for image in images:
                if image[-3:] == 'pts':
                    _, n_points = read_pts_file(f'{name}/{name_process}/{image}')
                    if n_points == 68:
                        dataset_names[name][name_process].append(images[:-4])
                    else:
                        continue
                else:
                    continue

    with open("dataset_names.json", "w") as f:
        json.dump(dataset_names, f, indent=4)

    print('Images name with 68 points saved to the dataset_names.json')