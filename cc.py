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
from fundus import Crawler
from fundus.publishers.base_objects import PublisherSpec, PublisherEnum  # noqa: F401
from fundus.publishers.de import DE  # noqa: F401

warnings.filterwarnings("ignore")

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


def fetch_wikinews():
    dsets = []
    print("Loading Wikinews dataset")
    ds = datasets.load_dataset(
        "malteos/wikinews",
        "de",
        download_mode="reuse_dataset_if_exists",
        cache_dir="volume_ds_cache",
        streaming=False,
    )
    splits = ds.keys()
    for split in splits:
        wn = ds[split]
        column_names = [col for col in wn.column_names if col != "cleaned_text"]
        wn = wn.rename_column("cleaned_text", "content")
        wn = wn.remove_columns(column_names=column_names)
        dsets.append(wn)

    wn = datasets.concatenate_datasets(dsets)
    print(wn)
    return wn


@filecache(365 * 24 * 60 * 60)
def fetch_fundus(collection):
    crawler = Crawler(collection)
    ds_data = {"content": []}
    for record in tqdm(crawler.crawl()):
        if record:
            ds_data["content"].append(str(record.body))
    ds = datasets.Dataset.from_dict(ds_data)
    return ds


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
    # "de.wikihow.com",
    # "correctiv.org",
    # "de.wikinews.org",
    # "scinexx.de",
    # "efahrer.chip.de",
]

fundus_specs = [prov for prov in DE]
urls += fundus_specs

for url in urls:
    data_dict = {"content": []}
    checkpointer = 0
    # Search the index for the target URL
    if url == "de.wikipedia.org":
        ds: datasets.Dataset = fetch_wikipedia()
        ds.push_to_hub(
            "flozi00/german_knowledge",
            split=url.replace("/", "").replace("*", ""),
            private=True,
            max_shard_size="5GB",
        )
        continue
    elif url == "de.wikinews.org":
        ds: datasets.Dataset = (
            fetch_wikinews()
            .filter(lambda x: len(x["content"]) >= 512)
            .cast_column("content", datasets.Value("string"))
        )
        ds.push_to_hub(
            "flozi00/german_knowledge",
            split=url.replace("/", "").replace("*", ""),
            private=True,
            max_shard_size="5GB",
        )
    elif isinstance(url, PublisherEnum):
        ds = fetch_fundus(url)
        ds.push_to_hub(
            "flozi00/german_knowledge",
            split=url.name,
            private=True,
            max_shard_size="5GB",
        )
    else:
        nlp = de_dep_news_trf.load()
        nlp.add_pipe("textdescriptives/information_theory")
        if "*" in url:
            records = search_cc_index(url)
            print("Searched Common Crawl Index")
        else:
            records = fetch_sitemap(url)
            print("Fetched sitemap")
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

                markdown = trafilatura.extract(
                    page,
                    target_language="de",
                    favor_precision=True,
                    favor_recall=True,
                    include_comments=False,
                )
                if markdown is None:
                    time.sleep(3)
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
                    if checkpointer % 1000 == 0:
                        ds = datasets.Dataset.from_dict(data_dict)
                        ds.push_to_hub(
                            "flozi00/german_knowledge",
                            split=url.replace("/", "").replace("*", ""),
                            private=True,
                        )
                else:
                    time.sleep(3)
        ds = datasets.Dataset.from_dict(data_dict).cast_column(
            "content", datasets.Value("string")
        )
        ds.push_to_hub(
            "flozi00/german_knowledge",
            split=url.replace("/", "").replace("*", ""),
            private=True,
            max_shard_size="5GB",
        )
