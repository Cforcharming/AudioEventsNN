from models import training


def main():
    
    # while True:
    #     info = input('Use database and model (split by ' '):')
    #     if info == 'exit':
    #         break
    #
    #     training.train(info)
    training.train('mnist mnist')


if __name__ == "__main__":
    main()
