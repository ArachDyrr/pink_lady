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
    plt.plot([epoch[0] for epoch in inputlist], label='training')
    plt.plot([epoch[1] for epoch in inputlist], label='validation')
    plt.legend()
    plt.ylim(ymin=0,ymax=1)
    plt.show(block=False)
    plt.pause(.1)

def startplot():
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin=0,ymax=1)
    plt.legend()
    plt.show()