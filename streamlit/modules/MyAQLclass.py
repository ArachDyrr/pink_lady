# Path: factory/modules/MyALQclass.py
# import numpy as np
# import pandas as pd


class MyAQLclass:
    def __init__(
        self,
        test_input=None,
        lotsize=500,
        product_class=1,
        test_inspection_lvl="I",
        first_max_percentage=0.4,
        secund_max_percentage=6.5,
        third_max_percentage=15,
    ):
        self.test_input = test_input
        self.lotsize = lotsize
        self.product_class = product_class
        self.test_inspection_lvl = test_inspection_lvl
        self.first_max_percentage = first_max_percentage
        self.secund_max_percentage = secund_max_percentage
        self.third_max_percentage = third_max_percentage

    def get_test_input(self):
        return self.test_input

    def get_lotsize(self):
        return self.lotsize

    def get_product_class(self):
        return self.product_class

    def get_test_inspection_lvl(self):
        return self.test_inspection_lvl

    def get_first_percentage(self):
        return self.first_max_percentage

    def get_secund_percentage(self):
        return self.secund_max_percentage

    def get_third_percentage(self):
        return self.third_max_percentage

    def get_reject_percentage(self):
        return self.third_max_percentage

    def table_B(codeLeter="F", inspection_lvl="I"):
        if codeLeter == "F" and inspection_lvl == "I":
            return 32
        else:
            return None

    def table_A(self, lotsize, test_inspection_lvl):
        if lotsize >= 281 and lotsize <= 500 and test_inspection_lvl == "I":
            code_letter = "F"
            return code_letter
        else:
            return None

    def table_sample_size(self, codeLeter, acceptablePercent):
        sample_size = 0
        if codeLeter == "F" and acceptablePercent <= 0.4:
            sample_size = 32
        elif codeLeter == "F" and acceptablePercent <= 6.5:
            sample_size = 20
        elif codeLeter == "F" and acceptablePercent <= 15:
            sample_size = 20
        else:
            sample_size = None

        return sample_size

    def rejectionValue(self, codeLeter, acceptablePercent):
        reject_nr = 0
        if codeLeter == "F" and acceptablePercent <= 0.4:
            reject_nr = 1
        elif codeLeter == "F" and acceptablePercent <= 6.5:
            reject_nr = 2
        elif codeLeter == "F" and acceptablePercent <= 15:
            reject_nr = 6
        else:
            reject_nr = None

        return reject_nr

    def first_rejectionValue(codeLeter, acceptablePercent):
        pass

    def batch_size(self):
        letter_code = self.table_A(self.lotsize, self.test_inspection_lvl)
        batch_size_first = self.table_sample_size(
            letter_code, self.first_max_percentage
        )
        batch_size_secund = self.table_sample_size(
            letter_code, self.secund_max_percentage
        )
        batch_size_third = self.table_sample_size(
            letter_code, self.third_max_percentage
        )
        batch_size = max([batch_size_first, batch_size_secund, batch_size_third])

        return batch_size

    def cut_off_per_class(self):
        letter_code = self.table_A(self.lotsize, self.test_inspection_lvl)
        reject_value_first = self.rejectionValue(letter_code, self.first_max_percentage)
        reject_value_secund = self.rejectionValue(
            letter_code, self.secund_max_percentage
        )
        reject_value_third = self.rejectionValue(letter_code, self.third_max_percentage)
        cut_off_per_class = [
            reject_value_first,
            reject_value_secund,
            reject_value_third,
        ]
        return cut_off_per_class

    def AQL_test_parameters(self):
        AQL_test_parameters = [self.batch_size()] + list(self.cut_off_per_class())
        return AQL_test_parameters

    def output(self):
        input = self.test_input
        class_cutoff = self.cut_off_per_class()
        if input == None:
            raise ValueError("Input is empty")
        elif input < class_cutoff[0]:
            return "I"
        elif input < class_cutoff[1]:
            return "II"
        elif input < class_cutoff[2]:
            return "III"
        else:
            return "Rejected"


def test(test_input=0):
    testcase = MyAQLclass()
    testcase.test_input = test_input
    x = testcase.output()
    print("#-----------------------------#")
    print(f"{x} should be 1 if test_input = 0")
    print("#-----------------------------#")
    print(f"test_input: {testcase.get_test_input()}")
    print(f"lotsize: {testcase.get_lotsize()}")
    print(f"product_class: {testcase.get_product_class()}")
    print(f"test_inspection_lvl: {testcase.get_test_inspection_lvl()}")
    print(f"first_max_percentage: {testcase.get_first_percentage()}")
    print(f"secund_max_percentage: {testcase.get_secund_percentage()}")
    print(f"third_max_percentage: {testcase.get_third_percentage()}")
    print(f"reject_percentage: {testcase.get_reject_percentage()}")
    print(f'code_letter: {testcase.table_A(500, "I")}')
    print(f'sample_size: {testcase.table_sample_size("F", 0.4)}')
    print(f'rejection_value: {testcase.rejectionValue("F", 0.4)}')
    print(f"batch_size: {testcase.batch_size()}")
    print(f"cut_off_per_class: {testcase.cut_off_per_class()}")
    print(f"AQL_test_parameters: {testcase.AQL_test_parameters()}")
    print(f"output: {testcase.output()}")

test()
if __name__ == "__main__":
    test()


bad_apples = 30
testimngcase = MyAQLclass()
testimngcase.test_input = bad_apples
x = testimngcase.output()
print("#-----------------------------#")
print(f"{x} is the result")