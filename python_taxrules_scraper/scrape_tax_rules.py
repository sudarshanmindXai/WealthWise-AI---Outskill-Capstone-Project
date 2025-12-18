import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def clean_text(text):
    if not text: return ""
    junk_phrases = [
        "Previous", "Next", "Compare Rule", 
        "Show Related Rules and Contents", "x", 
        "Income Tax Department", "Close",
        "javascript:return"
    ]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped: continue
        if stripped in junk_phrases: continue
        if "WebForm_" in stripped or "DeltaSP" in stripped: continue
        if stripped in [">", "<", "|", ">>", "<<"]: continue
        lines.append(stripped)
    return '\n'.join(lines)

def is_valid_iframe_content(text, rule_title):
    if not text: return False
    if "WebForm_OnSubmit" in text or "DeltaSPWebPartManager" in text: return False
    
    norm_title = rule_title.replace(" ", "").replace("-", "").lower()
    norm_text = text.replace(" ", "").replace("-", "").lower()
    
    if norm_title in norm_text: return True
    
    if len(text) < 100:
        keywords = ["omitted", "deleted", "repealed", "section", "rule", "substituted", "inserted"]
        if any(k in text.lower() for k in keywords): return True
        return False
    return True

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    
    print("--- STARTING BROWSER (V12 - Image Input Pagination) ---")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 15)
    
    output_file = "Income_Tax_Rules_Final.md"
    
    try:
        driver.get("https://incometaxindia.gov.in/pages/rules/income-tax-rules-1962.aspx")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Income Tax Rules 1962\n\n")

        current_page_num = 1
        
        while True:
            print(f"\n=== PROCESSING PAGE {current_page_num} ===")
            
            # Wait for list to load
            try:
                wait.until(EC.presence_of_all_elements_located((By.PARTIAL_LINK_TEXT, "Rule -")))
            except:
                print("Rules list taking long to appear...")
                time.sleep(2)

            # Capture the first rule name to detect page change later
            try:
                first_rule_element = driver.find_elements(By.PARTIAL_LINK_TEXT, "Rule -")[0]
                first_rule_name_before = first_rule_element.text
            except:
                first_rule_name_before = "Unknown"

            # Count Rules
            rules = driver.find_elements(By.PARTIAL_LINK_TEXT, "Rule -")
            num_rules = len(rules)
            print(f"Found {num_rules} rules on this page.")
            
            if num_rules == 0:
                print("No rules found. Ending.")
                break

            # --- RULE SCRAPING LOOP ---
            for i in range(num_rules):
                try:
                    rules = driver.find_elements(By.PARTIAL_LINK_TEXT, "Rule -")
                    if i >= len(rules): break
                    
                    rule_link = rules[i]
                    rule_title = rule_link.text
                    print(f"  > [{i+1}/{num_rules}] Opening: {rule_title}")
                    
                    driver.execute_script("arguments[0].click();", rule_link)
                    time.sleep(2.5) 
                    
                    # Iframe Content Hunting
                    found_content = False
                    final_text = ""
                    iframes = driver.find_elements(By.TAG_NAME, "iframe")
                    
                    for frame in iframes:
                        try:
                            driver.switch_to.frame(frame)
                            body = driver.find_element(By.TAG_NAME, "body")
                            text_candidate = body.text
                            if is_valid_iframe_content(text_candidate, rule_title):
                                final_text = text_candidate
                                found_content = True
                                driver.switch_to.default_content()
                                break 
                            driver.switch_to.default_content()
                        except:
                            driver.switch_to.default_content()

                    if found_content and final_text:
                        cleaned = clean_text(final_text)
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(f"## {rule_title}\n\n{cleaned}\n\n---\n\n")
                    else:
                        print(f"    [WARNING] No content found for {rule_title}")
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(f"## {rule_title}\n\n[Content Not Found]\n\n---\n\n")

                    try:
                        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        time.sleep(1)
                    except:
                        pass
                except Exception as e:
                    print(f"    Error processing rule {i}: {e}")
                    driver.switch_to.default_content()

            # --- PAGINATION (FIXED FOR INPUT TYPE=IMAGE) ---
            print("  > Page finished. Hunting for Next button...")
            
            try:
                next_btn = None
                
                # Priority 1: Title="Next Page" (Confirmed from your HTML)
                try:
                    next_btn = driver.find_element(By.XPATH, "//input[@title='Next Page']")
                except:
                    pass

                # Priority 2: ID ends with 'imgbtnNext' (Confirmed from your HTML)
                if not next_btn:
                    try:
                        next_btn = driver.find_element(By.CSS_SELECTOR, "input[id$='imgbtnNext']")
                    except:
                        pass

                # Priority 3: Alt text="Next"
                if not next_btn:
                    try:
                        next_btn = driver.find_element(By.XPATH, "//input[@alt='Next']")
                    except:
                        pass

                if next_btn:
                    # Check disabled class (aspNetDisabled)
                    btn_class = next_btn.get_attribute("class") or ""
                    if "aspNetDisabled" in btn_class or "disabled" in btn_class:
                        print("    Next button is disabled. End of Scraping.")
                        break

                    print("    Clicking Next Page...")
                    driver.execute_script("arguments[0].click();", next_btn)
                    
                    # Wait for the first rule to change (Page Load Verification)
                    print("    Waiting for new page to load...")
                    timeout = 0
                    while timeout < 20: # 20 seconds max wait
                        try:
                            new_rules = driver.find_elements(By.PARTIAL_LINK_TEXT, "Rule -")
                            if new_rules:
                                new_first = new_rules[0].text
                                if new_first != first_rule_name_before:
                                    print(f"    Success! Page {current_page_num + 1} loaded (First rule: {new_first}).")
                                    break
                        except:
                            pass
                        time.sleep(1)
                        timeout += 1
                    
                    if timeout >= 20:
                        print("    [WARNING] Timed out waiting for page change. Proceeding anyway.")
                    
                    current_page_num += 1
                else:
                    print("    [ERROR] Could not find the Next button even with HTML source fix.")
                    break

            except Exception as e:
                print(f"Pagination Error: {e}")
                break

    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")

    finally:
        driver.quit()
        print(f"Done. Check {output_file}")

if __name__ == "__main__":
    main()