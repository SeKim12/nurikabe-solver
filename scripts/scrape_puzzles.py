from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

BLACK = -1
WHITE = 0

def scraper(url):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # extract world size from title
    title = driver.find_element(By.TAG_NAME, 'h1')
    splits = title.text.split('x')
    nrows, ncols = int(splits[0][-1]), int(splits[1][0])

    solution = np.zeros((nrows, ncols))

    # first, click solve
    old_html = driver.page_source
    driver.find_element(By.XPATH, "//input[@type='button' and @value='Solve']").click()
    wait.until(lambda driver: driver.page_source != old_html)

    table = driver.find_element(By.ID, 'contain')
    i = 0
    while True:
        try:
            col = table.find_element(By.ID, f'c{i}')
            if col.text != ' ':
                solution.flat[i] = int(col.text)
            elif '255' in col.value_of_css_property('background-color'):
                solution.flat[i] = WHITE
            else:
                solution.flat[i] = BLACK
            i += 1
        except Exception as e:
            break 
    driver.quit()
    return {url: solution}


def janko_scraper(url):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    script_elem = driver.find_element(By.ID, 'data')
    html = script_elem.get_attribute('innerHTML')

    begin_tok = '[begin]'
    solution_tok = '[solution]'
    problem_tok = '[problem]'
    moves_tok = '[moves]'

    begin_ind = html.find(begin_tok)
    solution_ind = html.find(solution_tok)
    problem_ind = html.find(problem_tok)
    moves_ind = html.find(moves_tok)
    
    meta = html[begin_ind + len(begin_tok): problem_ind].strip()
    if 'rows' in meta:
        nrows, ncols = int(meta.split('rows ')[-1].split('\n')[0]), int(meta.split('cols ')[-1].split('\n')[0])
    else:
        size = int(meta.split('size ')[-1].split('\n')[0])
        nrows, ncols = size, size

    solution = np.zeros((nrows, ncols))

    row_texts = html[problem_ind + len(problem_tok): solution_ind].strip().split('\n')
    for r in range(len(row_texts)):
        row_text = row_texts[r].replace(' ', '')
        col_cnt = 0
        itr = 0
        while True:
            if row_text[itr].isnumeric():
                end = itr + 1
                while end < len(row_text) and row_text[end].isnumeric():
                    end += 1
                solution[r, col_cnt] = int(row_text[itr: end])
                itr = end - 1
            itr = itr + 1
            col_cnt = col_cnt + 1
            if col_cnt >= ncols:
                break

    soln_row_texts = html[solution_ind + len(solution_tok): moves_ind].strip().split('\n')
    for r in range(nrows):
        soln_row_text = soln_row_texts[r].replace(' ', '')
        for c in range(ncols):
            if solution[r, c] > 0: continue
            solution[r, c] = WHITE if soln_row_text[c] == '-' else BLACK
    driver.quit()
    return {url: solution}

# urls = [f'https://www.janko.at/Raetsel/Nurikabe/{i:04d}.a.htm' for i in range(1, 4)]
# for url in urls:
#     print(url)
#     print(janko_scraper(url))
# a = janko_scraper('https://www.janko.at/Raetsel/Nurikabe/0014.a.htm')
# print(a)

# pid_min = 1
# pid_max = 6670
# bs = 32

# combined = {}
# for i in tqdm(range(pid_min, pid_max + 1, bs)):
#     urls = [f'https://www.logicgamesonline.com/nurikabe/archive.php?pid={j}' for j in range(i, min(i + bs, pid_max + 1))]
#     results = Parallel(n_jobs=bs)(delayed(scraper)(url) for url in urls)
#     combined.update({k:v for d in results for k,v in d.items()})
#     # checkpoint
#     np.savez('nurikabe.npz', **combined)

pid_min = 1
pid_max = 1100
bs = 32

combined = {}
for i in tqdm(range(pid_min, pid_max + 1, bs)):
    urls = [f'https://www.janko.at/Raetsel/Nurikabe/{j:04d}.a.htm' for j in range(i, min(i + bs, pid_max + 1))]
    results = Parallel(n_jobs=bs)(delayed(janko_scraper)(url) for url in urls)
    combined.update({k:v for d in results for k,v in d.items()})
    # checkpoint
    np.savez('janko_nurikabe.npz', **combined)
