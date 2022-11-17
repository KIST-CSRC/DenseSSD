from niryo import Niryo
import argparse
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', help='calling function', default="")
    parser.add_argument('--vialStorage_loc', help='vial store', default=1)
    parser.add_argument('--stiring_loc', help='stiring location', default=1)
    parser.add_argument('--vialHolder_loc', help='vial holder location', default=1)
    parser.add_argument('--vialBack_loc', help='vial back holder location', default=1)
    parser.add_argument('--cuvetteStorage_loc', help='cuvette storage location', default=1)
    parser.add_argument('--cuvetteHolder_loc', help='cuvette holder location', default=1)

    return parser.parse_args(argv)


def main(args):
    niryo_one = Niryo()
    if args.action == 'set_calibrate':
        niryo_one.set_calibrate()
    elif args.action == 'move_vialStorage_to_stiring':  # new commend
        niryo_one.move_vialStorage_to_stiring(vial_loc=args.vialStorage_loc, stiring_loc=args.stiring_loc)
    elif args.action == 'move_stiring_to_vialHolder':  # new commend
        niryo_one.move_stiring_to_vialHolder(stiring_loc=args.stiring_loc, vialHolder_loc=args.vialHolder_loc)
    elif args.action == 'move_vialHolder_to_storageBack':  # new commend
        niryo_one.move_vialHolder_to_storageBack(vialHolder_loc=args.vialHolder_loc, vialBack_loc=args.vialBack_loc)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))