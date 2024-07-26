import json

results_path = "res.json"

with open(results_path, "r") as f:
    data = json.load(f)



def check_gosts(d):
    gosts = []
    key, h1, h2, t, misalginment = d.values()

    if misalginment < min(0.1 * t, 2):
        gosts.append((507, 'B'))
    elif misalginment < min(0.15 * t, 2):
        gosts.append((507, 'C'))
    elif misalginment < min(0.25 * t, 2):
        gosts.append((507, 'D'))
    else:
        gosts.append((507, 'None'))