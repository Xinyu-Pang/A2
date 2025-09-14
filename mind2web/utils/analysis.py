import os
import json

def calc_ave_num(benchmark):
    dist_path = os.path.join("data", f"{benchmark}_websites_dist.json")
    website_dist = json.load(open(dist_path, "r"))


    num_website = len(website_dist)
    num_task = 0
    for w in website_dist:
        num_task += int(website_dist[w])
    return num_website, num_task

def calc_domain_num(benchmark):
    dist_path = os.path.join("data", f"{benchmark}_websites_dist.json")
    website_dist = json.load(open(dist_path, "r"))

    domain_dist = []
    subdomain_dist = []
    for w in website_dist:
        exp_path = os.path.join("results/extracted_data", f"{benchmark}", f"{w}_tmp_examples.json")
        exp_data = json.load(open(exp_path, "r", encoding="utf-8"))[0]
        domain_dist.append(exp_data["domain"])
        subdomain_dist.append(exp_data["subdomain"])
    return domain_dist, subdomain_dist

def main():
    all_domain = {"domain": {}, "subdomain": {}}
    for benchmark in ["test_task", "test_website", "test_domain"]:
        # num_w, num_t = calc_ave_num(benchmark)
        # print(f"{benchmark}: {num_t} tasks, {num_w} websites, average {num_t/num_w} per website")

        domain_dist, subdomain_dist = calc_domain_num(benchmark)
        all_domain["domain"][benchmark] = domain_dist
        all_domain["subdomain"][benchmark] = subdomain_dist
        print(f"{benchmark}: {len(domain_dist)} domains, {len(subdomain_dist)} subdomains")

if __name__ == "__main__":
    main()