from validation_exception import ValidationException
from application import Application
import sys

def main():
    try:
        app = Application()
        app.preinitialization()
        app.run()
    except ValidationException as e:
        print(e)
        sys.exit(3)

if __name__ == '__main__':
    main()
