import numpy as np

true = True
false = False


class Data:
    __instance = None

    @staticmethod
    def get():
        if Data.__instance is None:
            Data()
        return Data.__instance

    def __init__(self):
        if Data.__instance is not None:
            raise Exception("This class is a singleton :)")
        else:
            Data.__instance = self

        self.data = ""


def add2queue(name, res, output):
    Data.get().data += check(name, res, output)


def check(name, res, output):
    close = np.allclose(res, output)

    if type(res) == np.ndarray:
        res = res.tolist()

    if type(res) == list:
        res = [i.tolist() if type(i) == np.ndarray else i for i in res]

    if type(output) == np.ndarray:
        output = output.tolist()

    if type(output) == list:
        output = [i.tolist() if type(i) == np.ndarray else i for i in output]

    # with open("types.txt", "a") as file:
    #     file.write(f"{name}: {type(res)}, {type(output)}\n")

    print(close)

    mark = "T" if close else "F"

    out = name + "|" + str(res) + "|" + str(output) + "|" + mark + "|"

    # print(out, end="")

    return out


def print_res():
    print(Data.get().data)
