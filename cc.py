import textdescriptives as td
import de_dep_news_trf
import requests
import json
from urllib.parse import quote_plus
import io
import warcio
from tqdm import tqdm
import datasets
import warnings
from trafilatura import extract

warnings.filterwarnings("ignore")

nlp = de_dep_news_trf.load()
nlp.add_pipe("textdescriptives/information_theory")

# The URL of the Common Crawl Index server
CC_INDEX_SERVER = "http://index.commoncrawl.org/"
# The Common Crawl index you want to query
INDEX_NAME = "CC-MAIN-2024-18"  # Replace with the latest index name


# Function to search the Common Crawl Index
def search_cc_index(url):
    encoded_url = quote_plus(url)
    index_url = f"{CC_INDEX_SERVER}{INDEX_NAME}-index?url={encoded_url}&output=json"
    response = requests.get(index_url)
    if response.status_code == 200:
        records = response.text.strip().split("\n")
        return [json.loads(record) for record in records]
    else:
        return None


# Function to fetch the content from Common Crawl
def fetch_page_from_cc(records):
    for record in records:
        offset, length = int(record["offset"]), int(record["length"])
        prefix = record["filename"].split("/")[0]
        s3_url = f'https://data.commoncrawl.org/{record["filename"]}'
        response = requests.get(
            s3_url, headers={"Range": f"bytes={offset}-{offset+length-1}"}
        )
        if response.status_code == 206:
            # Process the response content if necessary
            # For example, you can use warcio to parse the WARC record
            yield response.content
        else:
            print(f"Failed to fetch data: {response.status_code}")
            yield None


def fetch_wikipedia():
    wk = datasets.load_dataset(
        "olm/wikipedia",
        language="de",
        date="20240420",
        trust_remote_code=True,
        cache_dir="volume_ds_cache",
        split="train",
    )
    print("Loaded Wikipedia dataset")
    for entry in wk:
        yield entry["text"]


urls = [
    "flexikon.doccheck.com/*",
    # "de.wikipedia.org/*",
    # "tagesschau.de/*",
]

for url in urls:
    data_dict = {"content": []}
    checkpointer = 0
    # Search the index for the target URL
    if url == "de.wikipedia.org/*":
        records = ["dummy"]
    else:
        records = search_cc_index(url)
        print("Searched Common Crawl Index")
    print(f"Found {len(records)} records for {url}")
    if url == "de.wikipedia.org/*":
        contents = fetch_wikipedia()
    else:
        contents = fetch_page_from_cc(records)

    pbar = tqdm(contents)
    for content in pbar:
        if content:
            if url == "de.wikipedia.org/*":
                markdown = content
            else:
                with io.BytesIO(content) as stream:
                    for record in warcio.ArchiveIterator(stream):
                        page: str = record.content_stream().read()
                        markdown = extract(
                            page,
                            target_language="de",
                            favor_precision=True,
                            favor_recall=True,
                            include_comments=False,
                        )
            if markdown is None:
                continue
            markdown = markdown.split("\n")
            filtered = []
            for text in markdown:
                doc = nlp(text)
                stats = td.extract_df(doc).to_dict()
                metric = stats["entropy"][0]
                if metric >= 0.3:
                    filtered.append(text)
            markdown = "\n".join(filtered)

            if len(markdown) >= 512:
                data_dict["content"].append(markdown)
                checkpointer += 1
                pbar.set_description(f"{checkpointer} pages processed")
                if checkpointer % 1000 == 0:
                    ds = datasets.Dataset.from_dict(data_dict)
                    ds.push_to_hub(
                        "flozi00/german_knowledge",
                        split=url.replace("/", "").replace("*", ""),
                        private=True,
                    )
    ds = datasets.Dataset.from_dict(data_dict)
    ds.push_to_hub(
        "flozi00/german_knowledge",
        split=url.replace("/", "").replace("*", ""),
        private=True,
    )
