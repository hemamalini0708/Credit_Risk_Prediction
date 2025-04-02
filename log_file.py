import logging


def phase_1(acc):
    log = logging.getLogger(acc)
    log.setLevel("DEBUG")

    # Create a script for the data
    handler = logging.FileHandler(f"C:\\Users\\geeth\\PycharmProjects\\Credit_Card_Project\\file_loggings\\{acc}.log", mode='w')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log
