import os
import json
from Bio import Entrez
from datetime import datetime, timedelta

# 1. Configure Email (required by Entrez)
Entrez.email = "rijhwaninilesh@gmail.com"

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def generate_date_intervals(start_date, end_date, days):
    """
    Generate date intervals between a start and end date.
    Each interval is 'days' long, from [current, current+days).
    """
    date_format = "%Y/%m/%d"
    start_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)
    current_datetime = start_datetime
    date_intervals = []

    while current_datetime < end_datetime:
        date_intervals.append(current_datetime.strftime(date_format))
        current_datetime += timedelta(days=days)

    date_intervals.append(end_datetime.strftime(date_format))
    return date_intervals

def fetch_pmc_details(id_list):
    """
    Fetch full-text XML records for a list of PMC IDs (e.g., ['PMC12345', ...])
    from Entrez.efetch(db='pmc').

    Returns:
        A list of parsed records (Entrez.read(...) output).
    """
    ids = ",".join(id_list)
    handle = Entrez.efetch(db='pmc', id=ids, retmode='xml')
    records = Entrez.read(handle, validate=False)
    return records

def parse_front_metadata(front):
    """
    Extract authors, journal title, and publication year from the <front> portion
    of a PMC article XML record.

    Returns a dict with keys: "authors", "journal_title", "year".
    """
    data = {
        "authors": [],
        "journal_title": "",
        "year": ""
    }

    # (A) Journal Title
    journal_meta = front.get("journal-meta", {})
    try:
        jtg = journal_meta["journal-title-group"]  # dict
        data["journal_title"] = jtg["journal-title"]
    except KeyError:
        pass

    # (B) Article Meta
    article_meta = front.get("article-meta", {})

    # (B1) Publication Year
    pub_dates = article_meta.get("pub-date", [])
    if isinstance(pub_dates, dict):
        pub_dates = [pub_dates]  # unify
    for pd in pub_dates:
        if "year" in pd:
            data["year"] = pd["year"]
            break

    # (B2) Authors
    # Look for <contrib-group contrib-type="author"> or similar
    contrib_groups = article_meta.get("contrib-group", [])
    if isinstance(contrib_groups, dict):
        contrib_groups = [contrib_groups]

    collected_authors = []
    for cg in contrib_groups:
        # Some <contrib-group> might have an attribute or might not
        contrib_list = cg.get("contrib", [])
        if isinstance(contrib_list, dict):
            contrib_list = [contrib_list]
        for contrib in contrib_list:
            ctype = contrib.get("@contrib-type", "")  # e.g., "author"
            if ctype == "author" or not ctype:
                # parse the <name> sub-dict
                name_data = contrib.get("name", {})
                surname = name_data.get("surname", "")
                given_names = name_data.get("given-names", "")
                full_name = (surname + ", " + given_names).strip(", ")
                if full_name.strip():
                    collected_authors.append(full_name)
    data["authors"] = collected_authors

    return data

def extract_body_text(body_section):
    """
    Recursively extract text from 'body' or nested 'sec' elements
    in the PMC XML structure returned by Entrez.
    """
    text_content = []

    if isinstance(body_section, dict):
        # Possibly has 'p', 'sec' keys
        # 1) Paragraphs <p>
        if 'p' in body_section:
            paragraphs = body_section['p']
            if isinstance(paragraphs, list):
                for p in paragraphs:
                    if isinstance(p, str):
                        text_content.append(p)
                    elif isinstance(p, dict):
                        text_content.append(str(p))  # or parse further
            else:
                text_content.append(str(paragraphs))

        # 2) Sub-sections <sec>
        if 'sec' in body_section:
            subsections = body_section['sec']
            if isinstance(subsections, list):
                for sec_el in subsections:
                    text_content.append(extract_body_text(sec_el))
            else:
                text_content.append(extract_body_text(subsections))

    elif isinstance(body_section, list):
        # If body_section is a list of sections
        for item in body_section:
            text_content.append(extract_body_text(item))
    elif isinstance(body_section, str):
        text_content.append(body_section)

    # Join all extracted text
    return " ".join([txt.strip() for txt in text_content if txt.strip()])

###############################################################################
# MAIN SCRIPT: Searching + Fetching PMC Full-Text
###############################################################################

def main():
    # 0) Config
    search_query = "intelligence"
    start_date = "2014/01/01"
    end_date = "2014/03/24"
    days_interval = 200
    output_json = "pmc_fulltext.json"
    chunk_size = 500  # how many IDs per batch

    # 1) Generate intervals
    intervals = generate_date_intervals(start_date, end_date, days_interval)
    print("[INFO] date intervals:", intervals)

    data_list = []

    # 2) For each interval pair, search PMC
    for interval_start, interval_end in zip(intervals[:-1], intervals[1:]):
        print(f"[INFO] Searching from {interval_start} to {interval_end} in PMC...")

        search_handle = Entrez.esearch(
            db='pmc',
            term=search_query,
            mindate=interval_start,
            maxdate=interval_end,
            retmax=20000  # adjust as needed
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        pmc_id_list = search_results.get('IdList', [])
        print(f"[INFO] Found {len(pmc_id_list)} PMC IDs in this interval.")

        # 3) Fetch details in chunks
        for chunk_start in range(0, len(pmc_id_list), chunk_size):
            chunk_ids = pmc_id_list[chunk_start:chunk_start + chunk_size]
            print(f"[INFO] Fetching {len(chunk_ids)} records from PMC...")

            papers = fetch_pmc_details(chunk_ids)
            # 'papers' is typically a list of records

            # 4) Parse each record
            for record in papers:
                print(f"[Debug] Record: \n{json.dumps(record, indent=2)}")
                # Usually record['OAI-PMH']...
                try:
                    pmh = record.get("OAI-PMH", {})
                    if "GetRecord" not in pmh:
                        # might be no 'GetRecord' if not open-access
                        continue
                    rec = pmh["GetRecord"].get("record", {})
                    metadata = rec.get("metadata", {})
                    if "article" not in metadata:
                        # not open access or partial record
                        continue
                    article = metadata["article"]
                except KeyError:
                    continue

                front = article.get("front", {})
                body = article.get("body", {})
                # Some have <back>, <ref-list>, etc.

                # A) Parse front metadata
                front_meta = parse_front_metadata(front)
                journal_title = front_meta["journal_title"]
                authors = front_meta["authors"]
                pub_year = front_meta["year"]

                # B) Title
                #  Some articles store in <article-meta><title-group><article-title>
                #  We can often fetch from parse_front_metadata, but let's just demonstrate:
                title = ""
                article_meta = front.get("article-meta", {})
                if "title-group" in article_meta:
                    tg = article_meta["title-group"]
                    if "article-title" in tg:
                        title = tg["article-title"]

                # C) Abstract
                #   If article_meta has 'abstract'
                abstract_text = ""
                if "abstract" in article_meta:
                    abs_ = article_meta["abstract"]
                    if isinstance(abs_, list):
                        abstract_text = " ".join(str(x) for x in abs_)
                    elif isinstance(abs_, dict):
                        # might contain sub-structure
                        abstract_text = str(abs_)
                    else:
                        abstract_text = str(abs_)

                # D) Full Text (body)
                full_text = extract_body_text(body)

                data_list.append({
                    "pmc_id": pmh.get("header", {}).get("identifier", ""),  # e.g. "PMC123456"
                    "title": title,
                    "authors": authors,
                    "journal": journal_title,
                    "year": pub_year,
                    "abstract": abstract_text,
                    "full_text": full_text
                })

    # 5) Save to JSON
    print(f"[INFO] Total collected records: {len(data_list)}")
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(data_list, jf, indent=2)

    print(f"[INFO] Done. Saved data to {output_json}.")

if __name__ == "__main__":
    main()
