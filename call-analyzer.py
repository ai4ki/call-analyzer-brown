import email
import html2text
import imaplib
import openai
import pandas as pd
import re
import smtplib
import streamlit as st
import time

from bs4 import BeautifulSoup as bs
from email.message import EmailMessage
from email.utils import formatdate
from langdetect import detect
from scipy import spatial
from screening_utils import get_text_chunks
from typing import List

html_converter = html2text.HTML2Text()
html_converter.ignore_links = True

# Secrets
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")
EMAIL = st.secrets("EMAIL")
PASSWORD = st.secrets("PASSWORD")
IMAP_SERVER = st.secrets("IMAP_SERVER")
SMTP_SERVER = st.secrets("SMTP_SERVER")

openai.api_key = OPENAI_API_KEY

ABO_LIST = ["service.bund.de", "gsb.bund.de", "bundesanzeiger", "dfg.aktuell"]

# Streamlit settings
st.set_page_config(layout="wide", page_title="Newsletter Screening")

with open("./css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
        """
        <style>
            [data-testid="stHeader"]::before {
                content: "ai4ki";
                font-family: Arial, sans-serif;
                font-weight: bold;
                font-size: 40px;
                color: #dbd6d9;
                position: relative;
                left: 30px;
                top: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

cols = st.columns([4, 1, 6, 1, 6])

# Load model prompts
with open("./assets/positive_prompt_de.txt", "r", encoding="utf-8") as f:
    prompt_de = f.read()

with open("./assets/positive_prompt_en.txt", "r", encoding="utf-8") as f:
    prompt_en = f.read()


def get_mail_data(subfolder):

    imap_server = imaplib.IMAP4_SSL(IMAP_SERVER)
    imap_server.login(EMAIL, PASSWORD)
    imap_server.select(f'inbox/{subfolder}')

    _, data = imap_server.search('UTF-8', 'ALL')
    if not data:
        return []

    emails = {"total": None, "data": []}

    n_mails = 0
    for num in data[0].split():
        n_mails += 1
        email_content = ""
        _, msg_data = imap_server.fetch(num, '(RFC822)')
        email_msg = email.message_from_bytes(msg_data[0][1])
        if email_msg.get_content_type() == "text/html":
            payload_str = email_msg.get_payload(decode=True).decode('utf-8')
            email_content = html_converter.handle(payload_str)
        elif email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == 'text/plain':
                    charset = part.get_content_charset()
                    if charset is None:
                        email_content += part.get_payload()
                    else:
                        email_content += part.get_payload(decode=True).decode(charset)
                elif part.get_content_type() == 'text/html':
                    payload_str = part.get_payload(decode=True).decode('utf-8')
                    email_content += payload_str
        else:
            email_content = email_msg.get_payload()

        emails['data'].append({'subject': email_msg['Subject'],
                               'from': email_msg['From'],
                               'date': email_msg['Date'],
                               'content': email_content})

    emails["total"] = n_mails

    return emails


def send_email(subject, address, body):

    msg = EmailMessage()
    msg['Date'] = email.utils.formatdate()
    msg['Message-ID'] = email.utils.make_msgid(domain='ai4ki.org')
    msg['From'] = EMAIL
    msg['To'] = address
    msg['Subject'] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_SERVER, 465) as smtp:
        smtp.login(EMAIL, PASSWORD)
        smtp.send_message(msg)


def preprocess_nl(content, abo):

    text_pp = []
    df = pd.DataFrame()
    if abo == ABO_LIST[0]:
        lines = content.splitlines()
        for k, line in enumerate(lines):
            if line[:4] == "http":
                line_count = 0
                for ll in range(k + 1, k + 10):
                    line_count += 1
                    if lines[ll] == '':
                        text_pp.append(lines[k - 1:k + line_count])
                        break
        call_list = [[item[0], item[1], item[2:]] for item in text_pp]
        df = pd.DataFrame(call_list, columns=['call', 'url', 'client_info'])
    elif abo == ABO_LIST[1]:
        trunc_start = content.find("Bekanntmachung")
        trunc_end = content.find("3 Zuwendungsempfänger")
        if trunc_start != -1:
            if trunc_end != -1:
                core_text = content[trunc_start:trunc_end]
            else:
                core_text = content[trunc_start:]
            core_text = core_text.replace("#", "")
            core_text = core_text.replace("|", "")
            text_pp.append(core_text.strip())
        else:
            text_pp.append(content)
        snippets = get_text_chunks(text_pp[0], None)
        df = pd.DataFrame(snippets, columns=['call'])
    elif abo == ABO_LIST[2]:
        soup = bs(content, 'html.parser')
        table = soup.find('table')
        df = pd.read_html(str(table), encoding='utf-8', header=0)[0]
        df['url'] = [tag.get('href') if tag.has_attr('href') else "kein Link" for tag in table.find_all('a')]
        df.rename(columns={'Titel': 'call'}, inplace=True)
        df['call'] = df.call.apply(lambda x: x.replace("» ", ""))
    elif abo == ABO_LIST[3]:
        calls = "Ausschreibungen\r"
        trunc_idx_start = content.find(calls) + len(calls) - 1
        trunc_idx_end = content.find("DFG international\r")
        if trunc_idx_start != -1:
            core_text = content[trunc_idx_start:trunc_idx_end]
            lines = core_text.splitlines()
            urls = []
            call_text = ""
            url_flag = False
            for line in lines:
                if 'http' not in line:
                    if url_flag:
                        text_pp.append([call_text.strip(), urls])
                        urls = []
                        call_text = ""
                        url_flag = False
                    call_text += line + " "
                else:
                    http_idx = line.find("http")
                    url = line[http_idx:].replace(">", "")
                    urls.append(url.strip())
                    url_flag = True
        df = pd.DataFrame(text_pp, columns=['call', 'urls'])

    return df, text_pp


def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric="cosine",
) -> List[List]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def evaluate_calls(df, abo, query_de, query_en):

    query_embeddings = [
        openai.Embedding.create(input=query_de, engine='text-embedding-ada-002')['data'][0]['embedding'],
        openai.Embedding.create(input=query_en, engine='text-embedding-ada-002')['data'][0]['embedding']
    ]

    df['embeddings'] = df.call.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

    if abo == ABO_LIST[3]:
        distances = []
        for _, r in df.iterrows():
            if detect(r.call) == 'en':
                query_index = 1
            else:
                query_index = 0
            distances.append(distances_from_embeddings(query_embeddings[query_index], [r.embeddings],
                                                       distance_metric='cosine')[0])
        df['distances'] = distances
    else:
        df['distances'] = distances_from_embeddings(query_embeddings[0], df['embeddings'].values,
                                                    distance_metric='cosine')

    return df


def keyword_check(word):

    with open("negative_keywords.txt", "r", encoding="utf-8") as file:
        negative_keywords = file.read().lower().split(",")

    keyword_flag = True
    for keyword in negative_keywords:
        if keyword.strip() in word.lower():
            keyword_flag = False

    return keyword_flag


def check_float(value, low, high):
    if low <= value <= high:
        return True
    return False


def evaluate_gsb(title_score, text_score):

    score = 0.8*title_score + 0.2*text_score
    evaluation = ""
    if check_float(score, 0.0, 0.175):
        evaluation = "**Diese Bekanntmachung ist geeignet.**"
    elif check_float(score, 0.175, 0.19):
        evaluation = "**Diese Bekanntmachung ist möglicherweise geeignet.**"
    elif check_float(score, 0.19, 1.0):
        evaluation = "**Diese Bekanntmachung ist nicht geeignet.**"

    return evaluation


with cols[0]:
    st.markdown("### Institutsprofil")
    st.markdown(f"{prompt_de}")
    nl_abo = st.selectbox("**Wähle ein Abo:**", ABO_LIST)

    mail_data = get_mail_data(nl_abo)
    if mail_data:
        n_newsletters = mail_data["total"]
        newsletters = [mail_data["data"][nl]["date"] for nl in range(n_newsletters)]
        st.markdown(f"**{n_newsletters} Newsletter geladen**<br><br>", unsafe_allow_html=True)
        top_k = st.slider("**Maximale Anzahl an Treffern:**", 1, 10, 3)
        mail_results_to = st.text_input(label="**Ergebnisse mailen an:**", placeholder="person@example.org")
    else:
        st.markdown("**Sorry, beim Download der Newsletter ist etwas schief gegangen** :(")

with cols[2]:
    st.markdown(f"### Abo: {nl_abo}")
    nl_selected = st.selectbox("**Wähle einen Newsletter:**", newsletters, label_visibility="visible")
    nl_index = newsletters.index(nl_selected)

    nl_content = mail_data["data"][nl_index]["content"]
    nl_as_df, nl_call_list = preprocess_nl(nl_content, nl_abo)
    n_calls = len(nl_as_df.index)

    text_area = False
    if nl_abo == ABO_LIST[1]:
        text_area = True
        nl_display = nl_call_list[0]
        n_calls = 1
    elif nl_abo == ABO_LIST[3]:
        text_area = True
        nl_display = ""
        for n, call in enumerate(nl_call_list):
            nl_display += "Ausschreibung " + str(n+1) + "\n" + call[0] + "\n\n"

    st.markdown(f"**{n_calls} Ausschreibung(en):**")
    if text_area:
        st.text_area(label="Ausschreibungen:", value=nl_display, height=500, label_visibility="collapsed")
    else:
        st.dataframe(nl_as_df)

    st.markdown("")
    nl_analyze = st.button("Newsletter auswerten")

with cols[4]:
    if nl_analyze:
        alert = "**Dieser Newsletter enthält vermutlich keine passenden Ausschreibungen!**"
        st.markdown("### Auswertungsergebnis")
        with st.spinner(f"Auswertung von {nl_abo} läuft..."):
            dataframe = evaluate_calls(nl_as_df, nl_abo, prompt_de, prompt_en)
            hit_count = 1
            result = ""
            if nl_abo == ABO_LIST[0]:
                for idx, rows in dataframe.sort_values('distances', ascending=True).iterrows():
                    if hit_count == 1:
                        if rows.distances > 0.18:
                            result += alert + 2 * "\n"
                            st.markdown(alert)
                    if keyword_check(rows.call):
                        client_data = '<br>'.join(nl_call_list[idx])
                        st.markdown(
                            f"{client_data}<br>*Übereinstimmung: {(1.0 - rows.distances) * 100:.2f} %*",
                            unsafe_allow_html=True
                        )
                        result += f"{client_data}\n\n"
                    if hit_count == top_k:
                        break
                    hit_count += 1
            elif nl_abo == ABO_LIST[1]:
                title_dist = dataframe.iloc[0].distances
                title_snip = dataframe.iloc[0].call
                st.markdown(f"{title_snip[:200]}...<br>*Titelübereinstimmung: {(1.0 - title_dist) * 100:.2f} %*",
                            unsafe_allow_html=True)
                average_score = 0
                st.markdown("**Top-Textpassagen:**")
                for _, rows in dataframe.sort_values('distances', ascending=True).iterrows():
                    st.markdown(f"...{rows.call[:400]}...")
                    average_score += rows.distances
                    if hit_count == top_k:
                        break
                    hit_count += 1
                average_score /= top_k
                st.markdown(f"*Mittlere Textübereinstimmung: {(1.0 - average_score) * 100:.2f} %*")
                gsb_eval = evaluate_gsb(title_dist, average_score)
                st.markdown(gsb_eval)
                result = f"{gsb_eval}\n\n{title_snip}..."
            elif nl_abo == ABO_LIST[2]:
                for _, rows in dataframe.sort_values('distances', ascending=True).iterrows():
                    if hit_count == 1:
                        if rows.distances > 0.18:
                            result += alert + 2 * "\n"
                            st.markdown(alert)
                    if keyword_check(rows.call):
                        st.markdown(
                            f"{rows.call}<br>Vergabestelle: {rows.Behörde}<br>{rows.url}<br>\
                            *Übereinstimmung: {(1.0 - rows.distances) * 100:.2f} %*",
                            unsafe_allow_html=True
                        )
                        result += f"{rows.Behörde}\n{rows.call}\n{rows.url}\n\n"
                    if hit_count == top_k:
                        break
                    hit_count += 1
            elif nl_abo == ABO_LIST[3]:
                for _, rows in dataframe.sort_values('distances', ascending=True).iterrows():
                    if hit_count == 1:
                        if rows.distances > 0.17:
                            result += alert + 2 * "\n"
                            st.markdown(alert)
                    st.markdown(f"{rows.call[:150]}...<br>*Übereinstimmung: {(1.0 - rows.distances) * 100:.2f} %*<br>\
                                {rows.urls[0]}", unsafe_allow_html=True)
                    result += f"{rows.call[:150]}\n{rows.urls[0]}\n\n"
                    if hit_count == top_k:
                        break
                    hit_count += 1

            if mail_results_to:
                if re.match(r"[^@]+@[^@]+\.[^@]+", mail_results_to):
                    header = f"Die Auswertung von {nl_abo}, Ausgabe {nl_selected}, hat folgendes ergeben:"
                    email_body = header + 2 * "\n" + result
                    send_email('Newsletter-Auswertung', mail_results_to, email_body)
                    time.sleep(5)
