# -*- coding: utf-8 -*-
import asyncio
import json

import aiohttp
from bs4 import BeautifulSoup


async def fetch_html(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return None


async def parse_page(session, url):
    # print(f"Парсим данные с {url}...")
    html = await fetch_html(session, url)
    if html is None:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    data = []

    for person in soup.find_all('h5', class_='first_child'):
        name = person.text.strip()  # ФИО
        position = person.find_next('p').text.strip()  # Должность
        phone_tag = person.find_next('a', href=lambda href: href and 'tel:' in href)
        phone = phone_tag.text.strip() if phone_tag else 'Нет номера'  # Номер телефона
        data.append({"ФИО": name, "Должность": position, "Телефон": phone})

    return data


async def process_parse(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [parse_page(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

        data = [item for sublist in results for item in sublist]
        return data

# Запуск асинхронного процесса
# if __name__ == "__main__":
def parse():
    # Список URL для обработки
    urls = [
        'https://misis.ru/university/management/rukovoditeli-otdelov/',
        'https://misis.ru/university/management/direktora/',
        'https://misis.ru/university/management/rektorat/'
    ]

    data = asyncio.run(process_parse(urls))
    with open('data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        # print("done")
    # for entry in data:
    #     print(entry)
