from .application import Application


def main():
    app = Application()
    app.preinitialization()
    app.run()


if __name__ == '__main__':
    main()
