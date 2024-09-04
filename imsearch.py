import re
import io
import os
import json
import shutil
import random
import requests
from PIL import Image
from urllib import parse
import joblib
import contextlib
from tqdm.auto import tqdm
import hashlib


ALL = None
CREATIVE_COMMONS = "Any"
PUBLIC_DOMAIN = "Public"
SHARE_AND_USE = "Share"
SHARE_AND_USE_COMMECIALLY = "ShareCommercially"
MODIFY_SHARE_AND_USE = "Modify"
MODIFY_SHARE_AND_USE_COMMERCIALLY = "ModifyCommercially"

_licenses = [
    ALL,
    CREATIVE_COMMONS,
    PUBLIC_DOMAIN,
    SHARE_AND_USE,
    SHARE_AND_USE_COMMECIALLY,
    MODIFY_SHARE_AND_USE,
    MODIFY_SHARE_AND_USE_COMMERCIALLY,
]

_HEADERS = {
    "user-agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0"
}

_HEADERS_DOWNLOAD = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


@contextlib.contextmanager
def tqdm_parallel(tqdm_object):
    """Context manager to patch joblib to display tqdm progress bar"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def download(
    query,
    folder=".",
    max_urls=None,
    thumbnails=False,
    parallel=False,
    shuffle=False,
    remove_folder=False,
    license=ALL,
    safe_search=True,
):
    if thumbnails:
        urls = get_image_thumbnails_urls(query, license, safe_search)
    else:
        urls = get_image_urls(query, license, safe_search)

    if shuffle:
        random.shuffle(urls)

    if max_urls is not None and len(urls) > max_urls:
        urls = urls[:max_urls]

    if remove_folder:
        _remove_folder(folder)

    _create_folder(folder)
    if parallel:
        return _parallel_download_urls(urls, folder)
    else:
        return _download_urls(urls, folder)


def _download(url, folder):
    try:
        filename = hashlib.md5(url.encode("utf-8")).hexdigest()
        if os.path.exists("{}/{}.jpg".format(folder, filename)):
            return True

        response = requests.get(
            url,
            stream=True,
            timeout=10.0,
            allow_redirects=True,
            headers=_HEADERS_DOWNLOAD,
        )
        with Image.open(io.BytesIO(response.content)) as im:
            webp_filename = "{}/{}.webp".format(folder, filename)
            im.save(webp_filename, "WEBP")
            return True
    except:
        return False


def _download_urls(urls, folder):
    downloaded = 0
    for url in tqdm(urls):
        if _download(url, folder):
            downloaded += 1
    return downloaded


def _parallel_download_urls(urls, folder):
    downloaded = 0
    with tqdm_parallel(tqdm(total=len(urls))):
        with joblib.Parallel(n_jobs=os.cpu_count()) as parallel:
            results = parallel(joblib.delayed(_download)(url, folder) for url in urls)
            for result in results:
                if result:
                    downloaded += 1
    return downloaded


def get_image_urls(query, license, safe_search):
    token = _fetch_token(query)
    return _fetch_search_urls(query, token, license, safe_search)


def get_image_thumbnails_urls(query, license, safe_search):
    token = _fetch_token(query)
    return _fetch_search_urls(query, token, license, safe_search, what="thumbnail")


def _fetch_token(query, URL="http://duckduckgo.com/"):
    res = requests.get(f"{URL}?q={query}&t=h_&iax=images&ia=images")
    if res.status_code != 200:
        return ""
    match = re.search(r"vqd=\"([\d-]+)\"", res.text, re.M | re.I)
    if match is None:
        return ""
    return match.group(1)


def _fetch_search_urls(
    q, token, license, safe_search, URL="https://duckduckgo.com/", what="image"
):
    query = {
        "vqd": token,
        "q": q,
        "l": "us-en",
        "o": "json",
        "f": ",,,,,",
        "p": "1" if safe_search else "-1",
        "s": "100",
        "u": "bing",
    }
    if license is not None and license in _licenses:
        query["f"] = f",,,,,license:{license}"

    urls = []
    _urls, next = _get_urls(f"{URL}i.js", query, what)
    urls.extend(_urls)
    while next is not None:
        query.update(parse.parse_qs(parse.urlsplit(next).query))
        _urls, next = _get_urls(f"{URL}i.js", query, what)
        urls.extend(_urls)
    return urls


def _get_urls(URL, query, what):
    urls = []
    res = requests.get(
        URL,
        params=query,
        headers=_HEADERS,
    )
    if res.status_code != 200:
        return urls, None

    data = json.loads(res.text)
    for result in data["results"]:
        urls.append(result[what])
    return urls, data["next"] if "next" in data else None


def _remove_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)


def _create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)