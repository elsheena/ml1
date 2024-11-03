from model import SpaceTitanicModel
import fire 

def main():
    fire.Fire(SpaceTitanicModel)

if __name__ == "__main__":
    main()
    model = SpaceTitanicModel()
    model.train('../data/train.csv', 'Transported')
    model.predict('../data/test.csv')