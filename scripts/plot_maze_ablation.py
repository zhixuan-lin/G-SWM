import matplotlib.pyplot as plt
import os.path as osp
import json

def load_data(paths):
    data = {}
    for key, path in paths.items():
        with open(path, 'r') as f:
            raw_dict = json.load(f)
        data[key] = raw_dict['num_mean']
        
    return data

def plot(data):
    """
    :param data: dictionary
    :return:
    """
    for key, num_mean in data.items():
        plt.plot(num_mean, label=key)
    plt.xlabel('Time step')
    plt.ylabel('#Agent')
    plt.legend()
    
if __name__ == '__main__':
    exps = ['maze', 'maze_no_mu', 'maze_no_aoe', 'maze_no_sa']
    paths = {exp: osp.join('output', 'eval', exp, f'maze-{exp}.json') for exp in exps}
    paths = {exp: path for exp, path in paths.items() if osp.exists(path)}
    data = load_data(paths)
    plot(data)
    plt.savefig('output/eval/maze_ablation.png')
    print('Plot for the ablation experiment saved to "output/eval/maze_ablation.png"')
