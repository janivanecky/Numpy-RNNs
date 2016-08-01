import matplotlib.pyplot as plt

class Grapher:
    def __init__(self, label):
        plt.ylabel(label)
        plt.ion()
        plt.show()

    def update(self, keys, data, save_file = None):
        plt.plot(keys, data)
        plt.draw()
        
        if save_file != None and save_file.endswith('png'):
		    plt.savefig(save_file)
        plt.pause(0.001)