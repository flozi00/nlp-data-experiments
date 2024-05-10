import time
import textdescriptives as td
import de_dep_news_trf
import requests
import json
from urllib.parse import quote_plus
import io
import trafilatura.sitemaps
import warcio
from tqdm import tqdm
import datasets
import warnings
import trafilatura
from filecache import filecache


warnings.filterwarnings("ignore")

nlp = de_dep_news_trf.load()
nlp.add_pipe("textdescriptives/information_theory")

# The URL of the Common Crawl Index server
CC_INDEX_SERVER = "http://index.commoncrawl.org/"
# The Common Crawl index you want to query
INDEX_NAME = "CC-MAIN-2024-18"  # Replace with the latest index name


# Function to search the Common Crawl Index
@filecache(365 * 24 * 60 * 60)
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
@filecache(365 * 24 * 60 * 60)
def fetch_page_from_cc(record):
    offset, length = int(record["offset"]), int(record["length"])
    prefix = record["filename"].split("/")[0]
    s3_url = f'https://data.commoncrawl.org/{record["filename"]}'
    response = requests.get(
        s3_url, headers={"Range": f"bytes={offset}-{offset+length-1}"}
    )
    if response.status_code == 206:
        # Process the response content if necessary
        # For example, you can use warcio to parse the WARC record
        return response.content
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None


def fetch_wikipedia():
    wk = datasets.load_dataset(
        "olm/wikipedia",
        language="de",
        date="20240420",
        trust_remote_code=True,
        cache_dir="volume_ds_cache",
        split="train",
    )
    column_names = [col for col in wk.column_names if col != "text"]
    print("Loaded Wikipedia dataset")
    wk = wk.rename_column("text", "content")
    wk = wk.remove_columns(column_names=column_names)
    wk = wk.filter(lambda x: len(x["content"]) >= 1024)
    print(wk)
    return wk


# use trafilatura to fetch sitemap
def fetch_sitemap(domain):
    domain = f"https://{domain}/sitemap.xml"
    links = trafilatura.sitemaps.sitemap_search(domain)

    return links


urls = [
    # "flexikon.doccheck.com/*",
    # "de.wikipedia.org",
    # "tagesschau.de/inland/*",
    # "tagesschau.de/ausland/*",
    # "tagesschau.de/wirtschaft/*",
    # "tagesschau.de/wissen/*",
    # "de.wikinews.org/*",
    "de.wikihow.com",
]

for url in urls:
    data_dict = {"content": []}
    checkpointer = 0
    # Search the index for the target URL
    if url == "de.wikipedia.org":
        ds = fetch_wikipedia()
        ds.push_to_hub(
            "flozi00/german_knowledge",
            split=url.replace("/", "").replace("*", ""),
            private=True,
            max_shard_size="5GB",
        )
        continue
    else:
        if "*" in url:
            records = search_cc_index(url)
        else:
            records = fetch_sitemap(url)
        print("Searched Common Crawl Index")
        print(f"Found {len(records)} records for {url}")
        pbar = tqdm(records)
        for content in pbar:
            if content:
                markdown = None
                if "*" in url:
                    content = fetch_page_from_cc(content)
                    with io.BytesIO(content) as stream:
                        for record in warcio.ArchiveIterator(stream):
                            page: str = record.content_stream().read()
                else:
                    page = trafilatura.fetch_url(content)
                    time.sleep(1)

                markdown = trafilatura.extract(
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

                if len(markdown) >= 1024 and markdown not in data_dict["content"]:
                    data_dict["content"].append(markdown)
                    checkpointer += 1
                    pbar.set_description(f"{checkpointer} pages processed")
                    if checkpointer % 100 == 0:
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
        max_shard_size="5GB",
    )
