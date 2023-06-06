import matplotlib.pyplot as plt
from IPython import display
plt.ion()

def realtimeplot(inputlist):

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(inputlist)
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)