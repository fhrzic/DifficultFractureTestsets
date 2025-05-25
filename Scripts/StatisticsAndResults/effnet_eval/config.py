# effnet_eval/config.py

PRAUC_MACRO_AVERAGE = True
COMBINATION_SUBPLOT_MEAN = False
N_BOOTSTRAPS = 256

regions_dict = {
    "ANK": ["Ankle", 15],
    "CLA": ["Clavicle", 3],
    "ELB": ["Elbow", 6],
    "FIN": ["Finger", 10],
    "FOA": ["Forearm", 7],
    "FOO": ["Foot", 16],
    "HAN": ["Hand", 9],
    "KNE": ["Knee", 13],
    "LOL": ["Lower Leg", 14],
    "PEL": ["Pelvis", 11],
    "SHO": ["Shoulder", 4],
    "SKU": ["Skull",1],
    "SPI": ["Spine",2],
    "THI": ["Thigh", 12],
    "TOE": ["Toe", 17],
    "UPA": ["Upper Arm", 5],
    "WRI": ["Wrist", 8]
}

projections_dict = {
    "difficult": "difficult",
    "matched": "easy",
}

efficientnet_dict = {
    "efficientnet-b0": "B0",
    "efficientnet-b1": "B1",
    "efficientnet-b2": "B2",
    "efficientnet-b3": "B3",
    "efficientnet-b4": "B4",
    "efficientnet-b5": "B5",
    "efficientnet-b6": "B6",
    "efficientnet-b7": "B7"
}

colors = {
    "matched": 'forestgreen',
    "difficult": 'tomato'
}
