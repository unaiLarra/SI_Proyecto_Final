# S.I. Proyecto Final: Anti-spoofing
![image](https://github.com/user-attachments/assets/0e29d186-027c-4efa-bb2e-13ed092c301e)
## How to run
Install the required packages and run the following command from the root folder of the repository(preferably on a virtual environment).
```
flask --app ServerScripts/api.py run
```
Install [Godot](https://godotengine.org/), import the **project.godot** file inside the [anti-spoof-demo](https://github.com/unaiLarra/SI_Proyecto_Final/tree/main/anti-spoof-demo) folder and run the project from the editor by pressing the play button on the top right.

![image](https://github.com/user-attachments/assets/d5b20eef-3041-491c-bcaf-3d148ba833f9)

## CNN structure
```python
class NotsoDeepCNN(nn.Module):
    def __init__(self):
        super(NotsoDeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 300)
        self.fc2 = nn.Linear(300, 40)
        self.fc3 = nn.Linear(40, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
```
![image](https://github.com/user-attachments/assets/c8291f3c-2e47-44c3-9d1c-b4592ba3eae0)

## Tools used
[PyTorch](https://pytorch.org/),
[Flask](https://flask.palletsprojects.com/en/stable/),
[Godot](https://godotengine.org/),
[Bruno](https://www.usebruno.com/)(*Strong recomendation over postman!*),
[Numpy](https://numpy.org/),
[Large Crowdcollected Face Anti-Spoofing Dataset](https://www.kaggle.com/datasets/faber24/lcc-fasd),
[Photopea](https://www.photopea.com/)
