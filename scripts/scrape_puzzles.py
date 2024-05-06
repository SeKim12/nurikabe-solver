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

pid_min = 1
pid_max = 6670
bs = 32

combined = {}
for i in tqdm(range(pid_min, pid_max + 1, bs)):
    urls = [f'https://www.logicgamesonline.com/nurikabe/archive.php?pid={j}' for j in range(i, min(i + bs, pid_max + 1))]
    results = Parallel(n_jobs=bs)(delayed(scraper)(url) for url in urls)
    combined.update({k:v for d in results for k,v in d.items()})
    # checkpoint
    np.savez('nurikabe.npz', **combined)