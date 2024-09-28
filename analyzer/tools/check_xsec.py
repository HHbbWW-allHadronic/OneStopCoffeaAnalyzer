import requests
from analyzer.datasets.samples import DatasetRepo

def getSampleXSec(sample_name):
    url = 'https://xsecdb-xsdb-official.app.cern.ch/api/search'
    response = requests.post(url, json={'process_name': sample_name})
    print(response)
    print(response.text)
    if not response:
        return None
    return response.json()


def main():
    basic = requests.HTTPBasicAuth('user', 'pass')
    repo = DatasetRepo.getConfig()
    for dataset in repo:
        ds = repo[dataset]
        for sample in ds.samples:
            if not sample.cms_dataset_regex:
                continue
            n = sample.cms_dataset_regex.split('/')[1]
            x = getSampleXSec(n)
            print(f"{n} = {x}")


if __name__ == '__main__':
    main()
