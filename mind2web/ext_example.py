"""Process the mind2web data."""
import os
import ast
import json
from collections import Counter
from utils.data import load_json

def count_num_of_websites(data_dir, benchmark, output_path, output_path_pairs):
    """for different benchmarks (settings), count the number of samples for each website"""
    datas = load_json(data_dir, benchmark)

    # cal website distribution
    websites = [d["website"] for d in datas]
    dist_websites = Counter(websites)

    # log website: domain pairs
    if os.path.exists(output_path_pairs):
        with open(output_path_pairs, "r", encoding="utf-8") as f_in:
            website_domains = json.load(f_in)
    else:
        website_domains = {}
    for d in datas:
        if d["website"] not in website_domains:
            website_domains[d["website"]] = {"domain": d["domain"], "subdomain": d["subdomain"]}

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(dict(dist_websites), f_out, indent=4, ensure_ascii=False)

    with open(output_path_pairs, "w", encoding="utf-8") as f_out:
        json.dump(website_domains, f_out, indent=4, ensure_ascii=False)

def load_data_example(data_path):
    datas = json.load(open(data_path, "r"))
    data = datas[0]
    action = data["actions"][0]
    attributes_str = action["pos_candidates"][0]["attributes"]
    attributes = ast.literal_eval(attributes_str)
    print(type(attributes))

if __name__ == "__main__":
    data_dir = "data"
    benchmark = "test_domain"
    output_path = f"data/{benchmark}_websites_dist.json"
    output_path_pairs = f"data/website_domain_pairs.json"

    data_path = "results/test_website/tmp_extracted_examples/tripadvisor_tmp_examples.json"
    # count_num_of_websites(data_dir, benchmark, output_path, output_path_pairs)
    load_data_example(data_path)