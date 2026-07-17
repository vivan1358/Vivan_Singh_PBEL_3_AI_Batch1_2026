import config
from src import db
from src.gui import launch


def main():
    config.ensure_dirs()
    db.init_db()
    launch()


if __name__ == "__main__":
    main()
