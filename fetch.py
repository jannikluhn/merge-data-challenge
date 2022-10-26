import os
import errno
import json
import urllib.parse
import requests

CL_URL = ""

MERGE_SLOT = 4700013
SLOT_RANGE = 24 * 60 * 60 // 12

# OUTPUT_DIR = "./attestations/"
OUTPUT_DIR = "./headers/"


def get_beacon_header(slot):
    return get_cl(f"/eth/v1/beacon/headers/{slot}")


def get_validators(slot):
    return get_cl(f"/eth/v1/beacon/states/{slot}/validators")


def get_beacon_block(slot):
    return get_cl(f"/eth/v2/beacon/blocks/{slot}")


def get_attestations(slot):
    return get_cl(f"/eth/v1/beacon/blocks/{slot}/attestations")


def get_cl(path):
    url = urllib.parse.urljoin(CL_URL, path)
    res = requests.get(url)
    if res.status_code == 404:
        return None
    res.raise_for_status()
    return res.json()


if __name__ == "__main__":
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    start_slot = MERGE_SLOT - SLOT_RANGE
    end_slot = MERGE_SLOT + SLOT_RANGE
    for slot in range(start_slot, end_slot):
        print(f"{((slot - start_slot) / (end_slot - start_slot) * 100):.1f}%")
        path = os.path.join(OUTPUT_DIR, f"{slot}.json")
        if os.path.exists(path):
            print(f"skipping slot {slot} as file already exists")
            continue
        data = get_beacon_header(slot)
        with open(path, "x") as f:
            json.dump(data, f)
