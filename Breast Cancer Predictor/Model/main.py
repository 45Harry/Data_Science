



def main():
    #Create the model
    model = create_model()

    #Train the model
    train(model)

    #Evaluate the model
    evaluate(model)


if __name__ == '__main__':
    main()