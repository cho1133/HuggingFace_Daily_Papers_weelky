import os
import time
import requests
from lxml import html
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

def setup_api():
    """
    .env 파일에서 환경 변수를 불러오고 OpenAI 클라이언트를 설정합니다.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경 변수를 찾을 수 없습니다.")
        print(".env 파일이 존재하고, 그 안에 OPENAI_API_KEY가 올바르게 설정되었는지 확인해주세요.")
        return None
    
    print("OpenAI API 키를 성공적으로 불러왔습니다.")
    return OpenAI(api_key=api_key)

def get_previous_week_info():
    """
    오늘을 기준으로 지난주의 연도와 주차를 계산합니다.
    """
    today = datetime.now()
    last_week_date = today - timedelta(days=7)
    year, week, _ = last_week_date.isocalendar()
    return year, week

def translate_text_with_openai(client, text):
    """
    OpenAI API를 사용하여 주어진 텍스트를 한글로 번역합니다.
    연결 오류 시 몇 차례 재시도하는 로직이 포함되어 있습니다.
    """
    if not text or not text.strip():
        return "번역할 텍스트가 없습니다."
    if not client:
        return "OpenAI 클라이언트가 설정되지 않았습니다."
    
    max_retries = 3
    delay = 2  # 2초부터 시작

    for attempt in range(max_retries):
        try:
            # 사용자가 요청한 gpt-4o 모델 및 프롬프트로 변경
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a technical AI researcher. Please translate English academic papers into Korean. Please add a line break for each sentence."},
                    {"role": "user", "content": f"Translate the following English text to Korean:\n\n{text}"}
                ],
                temperature=0.1,
                timeout=20, # [수정됨] 응답 대기시간(초) 설정 추가
            )
            return response.choices[0].message.content.strip()
        # 구체적인 네트워크 예외를 잡도록 변경
        except (requests.exceptions.ConnectionError, OpenAI.APIConnectionError) as e:
            print(f"   > 번역 시도 {attempt + 1}/{max_retries} 실패: 네트워크 오류. {delay}초 후 재시도합니다.")
            time.sleep(delay)
            delay *= 2  # 다음 재시도 시 대기 시간 2배 증가
        except Exception as e:
            return f"번역 중 예상치 못한 오류 발생: {e}"

    return f"번역 중 오류 발생: {max_retries}번의 시도 후에도 네트워크 연결에 실패했습니다."


def scrape_and_translate_papers():
    """
    Hugging Face 지난주 인기 논문 10개의 정보를 스크레이핑하고,
    초록을 번역하여 파일로 저장하는 메인 함수입니다.
    """
    openai_client = setup_api()
    if not openai_client:
        return

    year, week = get_previous_week_info()
    week_str = f"{year}-W{week:02d}"
    
    target_url = f"https://huggingface.co/papers/week/{week_str}"
    output_filename = f"huggingface_top_10_papers_{week_str}_translated.txt"
    num_papers_to_scrape = 10
    base_url = "https://huggingface.co"

    print(f"'{target_url}'에서 논문 목록을 가져옵니다...")
    
    try:
        response = requests.get(target_url, timeout=10)
        response.raise_for_status()
        
        tree = html.fromstring(response.content)
        paper_articles = tree.xpath("/html/body/div[1]/main/div[2]/section/div[2]/article")[:num_papers_to_scrape]

        if not paper_articles:
            print(f"{week_str}에 해당하는 논문 정보를 찾을 수 없습니다.")
            return

        print(f"총 {len(paper_articles)}개의 논문을 찾았습니다. 파일에 저장을 시작합니다...")

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"Hugging Face 주간 인기 논문 Top 10 ({week_str}) - 번역본\n")
            f.write(f"출처: {target_url}\n\n")
            
            for i, article in enumerate(paper_articles):
                try:
                    h3_element = article.xpath(".//h3")[0]
                    link_list = h3_element.xpath("./a/@href")
                    title_list = h3_element.xpath("./a/text()")

                    if not link_list or not title_list:
                        raise ValueError("h3 태그 안에서 링크나 제목을 찾을 수 없습니다.")

                    relative_link = link_list[0]
                    title = "".join(title_list).strip()
                    paper_url = f"{base_url}{relative_link}"
                    
                    print(f"\n({i+1}/{len(paper_articles)}) '{title}' 처리 중...")
                    
                    f.write(f"--- {i+1}. {title} ---\n")
                    f.write(f"Link: {paper_url}\n\n")

                    paper_response = requests.get(paper_url, timeout=10)
                    paper_response.raise_for_status()
                    paper_tree = html.fromstring(paper_response.content)

                    abstract_elements = paper_tree.xpath('/html/body/div/main/div/section[1]/div/div[2]/div/p')
                    
                    if abstract_elements:
                        abstract = abstract_elements[0].text_content().strip()
                        print("   > 초록을 찾았습니다. 번역을 시작합니다...")
                        
                        f.write("## Abstract (Original)\n")
                        f.write(abstract + "\n\n")
                        
                        translated_abstract = translate_text_with_openai(openai_client, abstract)
                        print("   > 번역 완료.")
                        
                        f.write("## 초록 (Korean)\n")
                        f.write(translated_abstract + "\n\n")
                    else:
                        f.write("초록을 찾을 수 없습니다.\n\n")

                except Exception as e:
                    error_msg = f"항목 처리 중 오류 발생: {e}"
                    print(f"   > 오류: {error_msg}")
                    f.write(f"--- {i+1}. 항목 처리 실패 ---\n{error_msg}\n\n")

        print(f"\n완료! 모든 정보가 '{output_filename}' 파일에 저장되었습니다.")

    except requests.exceptions.RequestException as e:
        print(f"페이지를 가져오는 데 실패했습니다: {e}")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    scrape_and_translate_papers()
