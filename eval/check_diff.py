import argparse
import json

def main(args):
    with open(args.golden_file, 'r') as f:
        golden_data = f.readlines()
    with open(args.test_file, 'r') as f:
        test_data = f.readlines()

    assert len(golden_data) == len(test_data), "Files have different number of lines."

    tot_cnt = len(golden_data)
    one_one_cnt, one_zero_cnt, zero_one_cnt, zero_zero_cnt = 0, 0, 0, 0
    gold_empty_cnt, test_empty_cnt = 0, 0
    for i, (golden_line, test_line) in enumerate(zip(golden_data, test_data)):
        golden_json = json.loads(golden_line)
        test_json = json.loads(test_line)

        assert args.check_key in golden_json, f"Key {args.check_key} not found in golden file."
        assert args.check_key in test_json, f"Key {args.check_key} not found in test file."
        golden_value = golden_json.get(args.check_key, None)
        test_value = test_json.get(args.check_key, None)

        golden_value = int(golden_value)
        test_value = int(test_value)

        if golden_value == 1 and test_value == 1:
            one_one_cnt += 1
        elif golden_value == 1 and test_value == 0:
            one_zero_cnt += 1
        elif golden_value == 0 and test_value == 1:
            zero_one_cnt += 1
        elif golden_value == 0 and test_value == 0:
            zero_zero_cnt += 1

        if golden_json[args.print_key][0][0] == '':
            gold_empty_cnt += 1
        if test_json[args.print_key][0][0] == '':
            test_empty_cnt += 1

    print(f"Total lines: {tot_cnt}")
    print('golden : test')
    print(f'one count in golden: {one_one_cnt + one_zero_cnt}')
    print(f'one count in test: {one_one_cnt + zero_one_cnt}')
    print(f"Empty count in golden: {gold_empty_cnt}")
    print(f"Empty count in test: {test_empty_cnt}")
    print(f"1-1 count: {one_one_cnt}")
    print(f"1-0 count: {one_zero_cnt}")
    print(f"0-1 count: {zero_one_cnt}")
    print(f"0-0 count: {zero_zero_cnt}")

    # print ratios
    print(f"1-1 ratio: {one_one_cnt / tot_cnt:.4f}")
    print(f"1-0 ratio: {one_zero_cnt / tot_cnt:.4f}")
    print(f"0-1 ratio: {zero_one_cnt / tot_cnt:.4f}")
    print(f"0-0 ratio: {zero_zero_cnt / tot_cnt:.4f}")


    for i, (golden_line, test_line) in enumerate(zip(golden_data, test_data)):
        golden_json = json.loads(golden_line)
        test_json = json.loads(test_line)

        assert args.check_key in golden_json, f"Key {args.check_key} not found in golden file."
        assert args.check_key in test_json, f"Key {args.check_key} not found in test file."
        golden_value = golden_json.get(args.check_key, None)
        test_value = test_json.get(args.check_key, None)

        if golden_value != test_value:
            print('=' * 50)
            print(f"Difference found at line {i + 1}:")
            print(f'Full Golden Line: {golden_line.strip()}')
            print(f'Full Test Line: {test_line.strip()}')
            print(f"Golden {args.check_key}: {golden_value}")
            print(f"Test {args.check_key}: {test_value}")
            print(f'gold reps: ```\n{golden_json[args.print_key][0][0]}\n```')
            print(f'test reps: ```\n{test_json[args.print_key][0][0]}\n```')

            a = input("Continue checking? (y/n): ")
            if a.lower() == 'n':
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--check_key', type=str, required=True)
    parser.add_argument('--print_key', type=str, default='resps')
    args = parser.parse_args()

    main(args)
