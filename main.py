from models import training
from dataset import mivia_db


def main():
    
    while True:
        info = input('Use database and model (split by ' '):')
        if info == 'exit':
            break

        training.train(info)
    training.train('mnist mnist')
    # (x_train, y_train), (x_test, y_test) = mivia_db.load_data()
    # for i in x_train:
    #     print(i.numpy().shape)
    

if __name__ == "__main__":
    main()
