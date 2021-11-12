import numpy as np
import matplotlib.pyplot as plt
import json


data = {'x': [], 'y': []}

if __name__ == '__main__':
    with open('data.json', 'r') as f:
        raw = json.loads(f.read())

        for item in raw['SPY']:
            date = item['label'].replace(' ', '').split(',')
            y = date[1]
            m = date[0][:3]
            d = date[0][3:]
            strfmt = f'{d} {m} {y}'
            data['x'].append(strfmt)
            data['y'].append(item['close'])

    plt.plot(data['x'], data['y'], 'g')
    # plt.grid(True)
    plt.subplots_adjust(bottom=0.3)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False
    )
    plt.xticks(rotation=75)
    plt.show()
