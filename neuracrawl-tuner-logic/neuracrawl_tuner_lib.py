"""
neuracrawl-tuner library

This module contains all the core functionality for neuracrawl-tuner:
- Project management
- Sitemap URL extraction and analysis
- URL regex generation for exclusions
- Interesting URL selection
- CSS selector extraction for content cleaning
"""

import asyncio
import logging
import os
import re
import shutil
import sys
from collections import Counter
from logging import getLogger
from pathlib import Path

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from crawl4ai.html2text import CustomHTML2Text
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from markdowncleaner import CleanerOptions as MarkdownCleanerOptions
from markdowncleaner import MarkdownCleaner
from pydantic import BaseModel, Field

# =============================================================================
# Logging
# =============================================================================

logger = getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

# =============================================================================
# Utilities
# =============================================================================


def convert_url_to_file_name(url: str) -> str:
    filename = url.replace("https://", "").replace("http://", "").replace("/", "_")
    return filename


def batch_items[T](
    items: list[T],
    max_item_count_per_batch: int,
) -> list[list[T]]:
    batches = [
        items[x_index : x_index + max_item_count_per_batch]
        for x_index in range(0, len(items), max_item_count_per_batch)
    ]
    return batches


# =============================================================================
# LLM Utilities
# =============================================================================


async def call_structured_llm[T: BaseModel](
    system_prompt: str,
    output_model: type[T],
) -> T:
    system_message = SystemMessage(system_prompt)
    human_message = HumanMessage("Erledige die Aufgabe.")
    messages = [system_message, human_message]
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:50025",
        api_key="litellm-api-key-1234",
        model="vertex_ai/gemini-2.5-flash",
        timeout=120,
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(
        output_model,
        method="json_schema",
    )
    llm_output = await structured_llm.ainvoke(messages)
    return llm_output


MAX_CONCURRENT_BATCH_COUNT = 20


async def call_structured_llm_batch[T: BaseModel](
    system_prompts: list[str],
    output_model: type[T],
) -> list[T]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCH_COUNT)

    async def process_rate_limited(
        batch_index: int,
        system_prompt: str,
    ) -> T:
        async with semaphore:
            log_message = (
                f"Processing prompt ({batch_index + 1}/{len(system_prompts)})..."
            )
            logger.info(log_message)

            result = await call_structured_llm(system_prompt, output_model)

            log_message = f"Processed prompt ({batch_index + 1}/{len(system_prompts)})."
            logger.info(log_message)

            return result

    process_tasks = [
        process_rate_limited(x_index, x_system_prompt)
        for x_index, x_system_prompt in enumerate(system_prompts)
    ]

    all_results = await asyncio.gather(*process_tasks)
    return all_results


# =============================================================================
# Sub State Management
# =============================================================================


class SubStateDataPack(BaseModel):
    url: str = Field()
    raw_html: str = Field()
    cleaned_html: str = Field()
    raw_markdown: str = Field()
    feedback: str = Field()


class SubState(BaseModel):
    versions_folder_path: Path = Field()
    cloneable_result_folder_path: Path = Field()
    sub_state_data_packs: list[SubStateDataPack] = Field()


class SubStateManager:
    def get_current_versions(self, folder_path: Path) -> list[str]:
        versions: list[str] = []

        for x_folder_path in folder_path.iterdir():
            if x_folder_path.is_dir() and x_folder_path.name.startswith("v_"):
                versions.append(x_folder_path.name)

        sorted_versions = sorted(versions)
        return sorted_versions

    def get_latest_version(self, folder_path: Path) -> str:
        versions = self.get_current_versions(folder_path)
        if len(versions) == 0:
            return "v_000"
        latest_version = versions[-1]
        return latest_version

    def get_next_version(self, folder_path: Path) -> str:
        latest_version = self.get_latest_version(folder_path)
        if latest_version == "v_000" and not (folder_path / latest_version).exists():
            return "v_000"

        version_number = int(latest_version.split("_")[1])
        next_version_number = version_number + 1
        next_version = f"v_{next_version_number:03d}"
        return next_version

    def save_sub_state(self, sub_state: SubState) -> None:
        next_version = self.get_next_version(sub_state.versions_folder_path)
        version_folder_path = sub_state.versions_folder_path / next_version
        version_folder_path.mkdir(exist_ok=True, parents=True)

        sub_state_file_path = version_folder_path / "save_state.json"
        sub_state_json = sub_state.model_dump_json(indent=2)
        sub_state_file_path.write_text(sub_state_json)

        results_folder_path = version_folder_path / "results"
        results_folder_path.mkdir(exist_ok=True, parents=True)

        if sub_state.cloneable_result_folder_path.exists():
            for x_item in sub_state.cloneable_result_folder_path.iterdir():
                if x_item.is_file():
                    shutil.copy2(x_item, results_folder_path / x_item.name)
                elif x_item.is_dir():
                    shutil.copytree(
                        x_item, results_folder_path / x_item.name, dirs_exist_ok=True
                    )

        for x_index, x_sub_state_data_pack in enumerate(sub_state.sub_state_data_packs):
            sub_state_data_pack_folder_name = (
                f"{x_index:03d}_{convert_url_to_file_name(x_sub_state_data_pack.url)}"
            )
            sub_state_data_pack_folder_path = (
                version_folder_path / sub_state_data_pack_folder_name
            )
            sub_state_data_pack_folder_path.mkdir(exist_ok=True, parents=True)

            url_file_path = sub_state_data_pack_folder_path / "05_url.txt"
            url_file_path.write_text(x_sub_state_data_pack.url)

            raw_html_file_path = sub_state_data_pack_folder_path / "10_raw_html.html"
            raw_html_file_path.write_text(x_sub_state_data_pack.raw_html)

            cleaned_html_file_path = (
                sub_state_data_pack_folder_path / "20_cleaned_html.html"
            )
            cleaned_html_file_path.write_text(x_sub_state_data_pack.cleaned_html)

            raw_markdown_file_path = (
                sub_state_data_pack_folder_path / "30_raw_markdown.md"
            )
            raw_markdown_file_path.write_text(x_sub_state_data_pack.raw_markdown)

            feedback_file_path = sub_state_data_pack_folder_path / "40_feedback.txt"
            feedback_file_path.write_text(x_sub_state_data_pack.feedback)

    def get_sub_state(
        self, versions_folder_path: Path, version: str | None = None
    ) -> SubState:
        if version is None:
            version_folders = sorted([
                x_folder
                for x_folder in versions_folder_path.iterdir()
                if x_folder.is_dir() and x_folder.name.startswith("v")
            ])

            if not version_folders:
                raise ValueError(f"No version folders found in {versions_folder_path}")

            version_folder_path = version_folders[-1]
        else:
            version_folder_path = versions_folder_path / version

            if not version_folder_path.exists():
                raise ValueError(f"Version folder {version_folder_path} does not exist")

        sub_state_file_path = version_folder_path / "save_state.json"

        if not sub_state_file_path.exists():
            raise ValueError(f"Sub state file not found at {sub_state_file_path}")

        sub_state_json = sub_state_file_path.read_text(encoding="utf-8")
        sub_state = SubState.model_validate_json(sub_state_json)

        return sub_state


SUB_STATE_MANAGER = SubStateManager()


# =============================================================================
# HTML Cleaner
# =============================================================================


class Cleaner:
    PRECLEANING_EXCLUSION_CSS_SELECTORS = [
        "script",
        "style",
        "nav",
        "footer",
        "head",
        "hr",
    ]

    _markdown_cleaner_options: MarkdownCleanerOptions
    _markdown_cleaner: MarkdownCleaner

    def __init__(self):
        self._markdown_cleaner_options = MarkdownCleanerOptions(
            min_line_length=4,
            remove_short_lines=False,
            remove_whole_lines=False,
            remove_sections=False,
            remove_duplicate_headlines=False,
            remove_footnotes_in_text=True,
            contract_empty_lines=True,
            crimp_linebreaks=True,
        )
        self._markdown_cleaner = MarkdownCleaner(options=self._markdown_cleaner_options)

    def preclean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for x_selector in self.PRECLEANING_EXCLUSION_CSS_SELECTORS:
            for x_element in soup.select(x_selector):
                x_element.decompose()
        precleaned_html = soup.prettify()
        return precleaned_html

    def clean_html(self, html: str, exclusion_css_selectors: list[str]) -> str:
        soup = BeautifulSoup(html, "html.parser")
        all_selectors = (
            self.PRECLEANING_EXCLUSION_CSS_SELECTORS + exclusion_css_selectors
        )
        for x_selector in all_selectors:
            for x_element in soup.select(x_selector):
                x_element.decompose()
        cleaned_html = soup.prettify()
        return cleaned_html

    def convert_html_to_markdown(self, html: str) -> str:
        html_to_text_converter = CustomHTML2Text()
        html_to_text_converter.body_width = 0
        html_to_text_converter.ignore_links = False
        markdown = html_to_text_converter.handle(html)
        cleaned_markdown = self._markdown_cleaner.clean_markdown_string(markdown)
        return cleaned_markdown


CLEANER = Cleaner()


# =============================================================================
# Project Manager
# =============================================================================


class ProjectManager:
    selected_project: str | None = None

    def get_project_folder_path(self) -> Path:
        if self.selected_project is None:
            raise ValueError("No project selected.")
        project_folder_path = Path(os.getcwd(), "projects", self.selected_project)
        return project_folder_path

    def set_project(self, project_name: str) -> None:
        self.selected_project = project_name
        project_folder_path = self.get_project_folder_path()
        project_folder_path.mkdir(exist_ok=True, parents=True)

    def get_project(self) -> str:
        if self.selected_project is None:
            raise ValueError("No project selected.")
        return self.selected_project


PROJECT_MANAGER = ProjectManager()


# =============================================================================
# System Prompts
# =============================================================================

GENERAL_SYSTEM_PROMPT = """<Generell>
- neuracrawl
    - neuracrawl ist ein Webcrawler, welcher eine Ausgangsdomain bekommt und von dort dann aus deepcrawlt, also sich durch alle Links der Seite hangelt und immer weiter nach neuen Links sucht.
    - Er ist sehr gut darin, eine einzige Webseite sehr ausführlich zu crawlen.
    - URL Ausschließungen
        - Dabei schließt neuracrawl aber auch bestimmte URL Gruppen/Subpfade aus.
        - Zum Beispiel, kann es sein, dass wir bei einer Webseite alle Veranstaltungen oder Newsartikel ausschließen wollen, da wir die zum Beispiel nochmal getrennt über eine API strukturiert auslesen.
    - Markdown Extraktion
        - Dabei extrahiert er extrem sauberes Markdown, ohne Header, Footer, Cookiebanner, Werbeinhalten, etc.
        - Die Daten am Ende enthalten nur den reinen Inhalt der Webseite.
        - Dabei geht er subtraktiv vor, also entfernt alle Elemente, welche "Verschmutzungen" darstellen.
        - Dies ist immer ein Spiel zwischen "wir wollen alles entfernen, was nicht wirklicher Inhalt ist" und "wir wollen nichts entfernen, was zum wirklichen Inhalt gehört".
        - Unser Grundsatz ist, dass wir so nah wie möglich an den wirklichen Inhalt rankommen wollen, ohne dabei aber Informationen zu verlieren. Wir dürfen auf keinen Fall echte Informationen verlieren, egal, wo diese auf der Webseite stehen.
    - neuracrawl benötigt generell die folgenden Einstellungen:
        - Ausgangsdomain und erlaubt andere Domains, auf welche er kommen und crawlen darf.
        - Ausschließ-URL-Regexes, welche bestimmte URL Gruppen/Subpfade ausschließen.
            - Zum Beispiel : "^.*/(aktuelles|amtsblatt)/.*$" oder "^.*\\.(?:ics|pdf).*$"
        - Ausschließ-CSS-Selektoren, welche bestimmte HTML-Elemente auf allen Seiten ausschließen.
            - Zum Beispiel : "header", ".front-left" oder "#cc-size"

- neuracrawl tuner ist eine Sammlung an Funktionen, welche dabei helfen, die perfekten Werte für die obigen Einstellungen zu finden.
</Generell>"""


# =============================================================================
# Sitemap URLs
# =============================================================================

# --- File helpers ---


def get_sitemap_urls_folder_path() -> Path:
    project_folder_path = PROJECT_MANAGER.get_project_folder_path()
    folder_path = project_folder_path / "05_sitemap_urls"
    folder_path.mkdir(exist_ok=True, parents=True)
    return folder_path


def get_sitemap_urls_txt_file_path() -> Path:
    sitemap_urls_folder_path = get_sitemap_urls_folder_path()
    file_path = sitemap_urls_folder_path / "sitemap_urls.txt"
    return file_path


def create_sitemap_urls_file() -> None:
    file_path = get_sitemap_urls_txt_file_path()
    if not file_path.exists():
        file_path.touch()


def save_sitemap_urls(urls: list[str]) -> None:
    file_path = get_sitemap_urls_txt_file_path()
    urls_text = "\n".join(urls)
    with open(file_path, "w") as file:
        file.write(urls_text)


def get_saved_sitemap_urls() -> list[str]:
    file_path = get_sitemap_urls_txt_file_path()
    with open(file_path, "r") as file:
        urls = [
            x_line.strip() for x_line in file.readlines() if len(x_line.strip()) > 0
        ]
    return urls


def get_frequent_sitemap_urls_txt_file_path() -> Path:
    sitemap_urls_folder_path = get_sitemap_urls_folder_path()
    file_path = sitemap_urls_folder_path / "frequent_sitemap_urls.txt"
    return file_path


def save_frequent_sitemap_urls(frequent_urls_text: str) -> None:
    frequent_urls_txt_file_path = get_frequent_sitemap_urls_txt_file_path()
    with open(frequent_urls_txt_file_path, "w") as file:
        file.write(frequent_urls_text)


def get_saved_frequent_sitemap_urls() -> str:
    file_path = get_frequent_sitemap_urls_txt_file_path()
    with open(file_path, "r") as file:
        frequent_urls_text = file.read()
    return frequent_urls_text


def get_sitemap_url_extensions_txt_file_path() -> Path:
    sitemap_urls_folder_path = get_sitemap_urls_folder_path()
    file_path = sitemap_urls_folder_path / "sitemap_url_extensions.txt"
    return file_path


def save_sitemap_url_extensions(url_extensions_text: str) -> None:
    file_path = get_sitemap_url_extensions_txt_file_path()
    with open(file_path, "w") as file:
        file.write(url_extensions_text)


def get_saved_sitemap_url_extensions() -> str:
    file_path = get_sitemap_url_extensions_txt_file_path()
    with open(file_path, "r") as file:
        url_extensions_text = file.read()
    return url_extensions_text


# --- Methods ---


def extract_sitemap_urls() -> None:
    urls_txt_file_path = get_sitemap_urls_txt_file_path()
    url_regex = r"https?://[^\s<>\"']+"
    with open(urls_txt_file_path, "r") as urls_txt_file:
        text = urls_txt_file.read()
    urls = re.findall(url_regex, text)
    save_sitemap_urls(urls)


def extract_frequent_sitemap_urls(min_frequency: int) -> None:
    """Extract common URL paths or path segments that appear frequently across URLs."""
    urls = get_saved_sitemap_urls()

    path_segments = []

    for x_url in urls:
        if "://" in x_url:
            url_without_protocol = x_url.split("://", 1)[1]
            if "/" in url_without_protocol:
                domain_and_path = url_without_protocol.split("/", 1)
                if len(domain_and_path) > 1:
                    path = domain_and_path[1]
                    parts = path.rstrip("/").split("/")
                    for i in range(1, len(parts) + 1):
                        segment = "/".join(parts[:i]) + "/"
                        path_segments.append(segment)

    segment_counter = Counter(path_segments)

    common_segments = {
        segment: count
        for segment, count in segment_counter.items()
        if count >= min_frequency
    }

    sorted_segments = sorted(common_segments.items(), key=lambda x: (-x[1], x[0]))

    frequent_sitemap_urls_text = "\n".join([
        f"{x_count}\t{x_segment}" for x_segment, x_count in sorted_segments
    ])
    save_frequent_sitemap_urls(frequent_sitemap_urls_text)

    print(
        f"Found {len(sorted_segments)} common URL areas with frequency >= {min_frequency}"
    )


def extract_url_extensions() -> None:
    urls = get_saved_sitemap_urls()

    url_extensions: list[str] = []

    for x_url in urls:
        if "." in x_url:
            url_parts = x_url.split("/")
            last_part = url_parts[-1]
            if "." in last_part:
                extension_parts = last_part.split(".")
                extension = extension_parts[-1]
                if "?" in extension:
                    extension = extension.split("?")[0]
                if "#" in extension:
                    extension = extension.split("#")[0]
                if len(extension) > 0 and len(extension) <= 10:
                    url_extensions.append(extension)
            else:
                url_extensions.append("html")
        else:
            url_extensions.append("html")

    counter = Counter(url_extensions)
    sorted_url_extensions = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    url_extensions_text = "\n".join([
        f"{x_count}\t{x_url_extension}"
        for x_url_extension, x_count in sorted_url_extensions
    ])
    save_sitemap_url_extensions(url_extensions_text)


# =============================================================================
# URL Regexes
# =============================================================================

# --- Prompts ---

URL_REGEXES_EXTRACTION_SYSTEM_PROMPT = """<Prozess>
- Du bist Teil des folgenden Prozesses:
    - Um die Ausschließ-URL-Regexes zu bestimmen, wird die Liste der häufigsten URL-Pfade einer Webseite analysiert.
    - Diese Liste enthält nur URLs, welche eine bestimmte Mindestanzahl an Unterseiten haben (z. B. mindestens 5), wodurch sie besonders relevant für Ausschließungen sind, da der Verdacht auf strukturierte, CMS Daten nahe liegt.
    - Das Ziel ist es, Regexes zu generieren, welche URLs markieren, welche ausgeschlossen werden sollten, da sie anderarbeitig oder gar nicht extrahiert werden sollen.
    - Die Regexes sollten präzise sein und nur die gewünschten URL-Gruppen matchen, ohne versehentlich wichtige Seiten auszuschließen.
    - Sie sollen aber auch so allgemein wie möglich gehalten werden, und nicht nur auf einzelne Seiten abzielen, sondern immer eher auf die Parent-Pfade, um z. B. alle Detailseiten zu erwischen.
    - Regexes sollten wenn möglich eine gesamte URL matchen, also von "^" bis "$". Oft haben die finalen URLs aber auch noch Queryparameter, deswegen achte auch gerade bei Dateiformaten auf diese, da die Regexes sonst nicht greifen.
    - Beispiele für typische Ausschließungen:
        - Veranstaltungskalender: "^.*/(veranstaltungen|termine|events)/.*$"
        - News/Aktuelles: "^.*/(aktuelles|news|artikel)/.*$"
        - Dateiformate: "^.*\\.(?:pdf|ics|xml|json)(?:\\?.*)?$"
        - Archive: "^.*/(archiv|archive)/.*$"
    - Standardmäßig soll nichts ausgeschlossen werden, auch die obigen Beispiele nicht. Der Nutzer gibt an, was er ausschließen möchte und nur das sollte dann auch ausgeschlossen werden.
    - Jedes Regex soll ordentlich begründet werden, um es nachvollziebar zu machen.
</Prozess>"""

URL_REGEXES_EXTRACTION_MAIN_SYSTEM_PROMPT = """
<Häufige URL-Pfade>
- Hier sind die häufigsten URL-Pfade mit ihrer Anzahl an Unterseiten:
{frequent_urls_text}
</Häufige URL-Pfade>

<URL Endungen>
- Hier sind alle URL Endungen mit ihrer Anzahl:
{url_extensions_text}
</URL Endungen>

<Zusatzanweisungen>
- Hier sind die spezifischen Nutzeranweisungen, welche du beachten sollst:
{custom_instructions}
</Zusatzanweisungen>"""


# --- LLM Output models ---


class UrlExclusionRegexSingleResultLlmOutput(BaseModel):
    url_regex: str = Field()
    reason: str = Field()


class UrlExclusionRegexesFullResultLlmOutput(BaseModel):
    url_regex_infos: list[UrlExclusionRegexSingleResultLlmOutput] = Field()


# --- System prompt methods ---


def generate_url_regexes_system_prompt(
    frequent_sitemap_urls_text: str,
    sitema_url_extensions_text: str,
    custom_instructions: str,
) -> str:
    parts: list[str] = []
    parts.append(GENERAL_SYSTEM_PROMPT)
    parts.append(URL_REGEXES_EXTRACTION_SYSTEM_PROMPT)

    main_system_prompt = URL_REGEXES_EXTRACTION_MAIN_SYSTEM_PROMPT.format(
        frequent_urls_text=frequent_sitemap_urls_text,
        url_extensions_text=sitema_url_extensions_text,
        custom_instructions=custom_instructions,
    )
    parts.append(main_system_prompt)

    system_prompt = "\n\n".join(parts)
    return system_prompt


# --- File helpers ---


def get_url_regexes_folder_path() -> Path:
    project_folder_path = PROJECT_MANAGER.get_project_folder_path()
    folder_path = project_folder_path / "10_excluded_urls"
    folder_path.mkdir(exist_ok=True, parents=True)
    return folder_path


def get_url_regexes_results_folder_path() -> Path:
    url_regexes_folder_path = get_url_regexes_folder_path()
    results_folder_path = url_regexes_folder_path / "results"
    results_folder_path.mkdir(exist_ok=True, parents=True)
    return results_folder_path


def get_url_regexes_file_path() -> Path:
    url_regexes_results_folder_path = get_url_regexes_results_folder_path()
    file_path = url_regexes_results_folder_path / "url_regexes.txt"
    return file_path


def save_url_regexes(url_regexes: list[str]) -> None:
    file_path = get_url_regexes_file_path()
    regexes_text = "\n".join(url_regexes)
    with open(file_path, "w") as file:
        file.write(regexes_text)


def get_saved_url_regexes() -> list[str]:
    file_path = get_url_regexes_file_path()
    with open(file_path, "r") as file:
        regexes = [
            x_line.strip() for x_line in file.readlines() if len(x_line.strip()) > 0
        ]
    return regexes


def get_url_regexes_reason_file_path() -> Path:
    url_regexes_results_folder_path = get_url_regexes_results_folder_path()
    file_path = url_regexes_results_folder_path / "url_regexes_reasons.txt"
    return file_path


def save_url_regexes_reason(
    llm_output: UrlExclusionRegexesFullResultLlmOutput,
) -> None:
    file_path = get_url_regexes_reason_file_path()
    reason = llm_output.model_dump_json(indent=2)
    with open(file_path, "w") as file:
        file.write(reason)


def get_excluded_urls_file_path() -> Path:
    url_regexes_results_folder_path = get_url_regexes_results_folder_path()
    file_path = url_regexes_results_folder_path / "excluded_urls.txt"
    return file_path


def save_excluded_urls(urls: list[str]) -> None:
    file_path = get_excluded_urls_file_path()
    urls_text = "\n".join(urls)
    with open(file_path, "w") as file:
        file.write(urls_text)


def get_saved_excluded_urls() -> list[str]:
    file_path = get_excluded_urls_file_path()
    with open(file_path, "r") as file:
        urls = [
            x_line.strip() for x_line in file.readlines() if len(x_line.strip()) > 0
        ]
    return urls


def get_non_excluded_urls_file_path() -> Path:
    url_regexes_results_folder_path = get_url_regexes_results_folder_path()
    file_path = url_regexes_results_folder_path / "non_excluded_urls.txt"
    return file_path


def save_non_excluded_urls(urls: list[str]) -> None:
    file_path = get_non_excluded_urls_file_path()
    urls_text = "\n".join(urls)
    with open(file_path, "w") as file:
        file.write(urls_text)


def get_saved_non_excluded_urls() -> list[str]:
    file_path = get_non_excluded_urls_file_path()
    with open(file_path, "r") as file:
        urls = [
            x_line.strip() for x_line in file.readlines() if len(x_line.strip()) > 0
        ]
    return urls


# --- Methods ---


async def extract_url_regexes(custom_instructions: str) -> None:
    log_message = "Extracting URL regexes..."
    logger.info(log_message)

    frequent_sitemap_urls_text = get_saved_frequent_sitemap_urls()

    sitemap_url_extensions_text = get_saved_sitemap_url_extensions()

    log_message = "Analyzing frequent URL paths..."
    logger.info(log_message)

    system_prompt = generate_url_regexes_system_prompt(
        frequent_sitemap_urls_text, sitemap_url_extensions_text, custom_instructions
    )
    full_result_llm_output = await call_structured_llm(
        system_prompt,
        UrlExclusionRegexesFullResultLlmOutput,
    )
    url_regexes = [
        x_regex_info.url_regex
        for x_regex_info in full_result_llm_output.url_regex_infos
    ]

    log_message = f"Found {len(url_regexes)} URL regexes."
    logger.info(log_message)

    save_url_regexes(url_regexes)
    save_url_regexes_reason(full_result_llm_output)

    log_message = f"Extracted {len(url_regexes)} URL regexes."
    logger.info(log_message)

    log_message = "Applying URL regexes to sitemap URLs..."
    logger.info(log_message)

    all_sitemap_urls = get_saved_sitemap_urls()

    excluded_urls: list[str] = []
    non_excluded_urls: list[str] = []

    for x_url in all_sitemap_urls:
        is_excluded = False
        for y_regex in url_regexes:
            if re.match(y_regex, x_url):
                is_excluded = True
                break
        if is_excluded:
            excluded_urls.append(x_url)
        else:
            non_excluded_urls.append(x_url)

    log_message = f"Applied URL regexes. Found {len(excluded_urls)} excluded and {len(non_excluded_urls)} non-excluded URLs."
    logger.info(log_message)

    save_excluded_urls(excluded_urls)
    save_non_excluded_urls(non_excluded_urls)

    log_message = "Saved excluded and non-excluded URLs."
    logger.info(log_message)


# =============================================================================
# Interesting URLs
# =============================================================================

# --- Prompts ---

INTERESTING_URLS_EXTRACTION_BATCH_SIZE = 500

INTERESTING_URLS_EXTRACTION_COMMON_SYSTEM_PROMPT = """<Prozess>
- Du bist Teil des folgenden Prozesses:
    - Um die CSS-Ausschließ-Selektoren zu bestimmen, müssen einige Sample Seiten der Webseite analysiert werden und auf diesen dann die CSS-Selektoren angewendet werden um zu schauen, ob sie den gewünschten Effekt haben.
    - Dazu werden zu erst aus der Sitemap einer Webseite interessante, diverse URLs ausgewählt, welche das Sample Set darstellen.
    - Dabei sollten diese Seiten besonders repräsentativ für die gesamte Webseite sein. Z. B. einmal die Startseite, dann eine Veranstaltungs-Übersichts-Seite, eine Veranstaltungs-Detail-Seite, eine News-Übersichts-Seite, eine News-Detail-Seite, eine Blog-Übersichts-Seite, eine Blog-Detail-Seite, eine Archivseite, eine Kontaktseite, eine Impressumseite, etc. Einfach die verschiedensten Datenstrukturen, Formate und Inhalte
    - Also ein Set an Seiten, bei dem wir auch unterschiedliche Inhalte und Layoutstrukturen erwarten.
    - Natürlich können wir das nicht genau wissen, da wir nur die URLs sehen und aus diesen einfach von außen auswählen müssen. Tortzdem lässt sich an den URLs und Pfadsegmenten schon sehr gut ablesen, welche Seiten unterschiedliche Inhalte enthalten sollten.
    - Da eine Webseite tausende Seiten enthalten kann, gehen wir hierbei in Batches vor. Zuerst extrahieren mehrere KI-Agenten aus jeweils 500 URLs ein Sample Set und begründen ihre Auswahlen. Es ist wichtig eine gute Begründung zu geben, damit der zusammenfassende KI-Agent die Gedanken hinter den Auswahlen besser versteht.
    - Dann nimmt ein zweiter KI-Agent die Batches und kombiniert diese zu einem finalen Sample Set, indem er versucht die besten URLs auszuwählen. Dabei versucht er auf maximal 15 URLs zu kommen.
    - Das Sample Set wird dann später heruntergeladen und vom Nutzer analysiert.
</Prozess>"""

INTERESTING_URLS_EXTRACTION_BATCH_SYSTEM_PROMPT = """<Aufgabe>
- Genauer gesagt, bist du der KI-Agent, welcher die Auswahl der URLs für das Sample Set vornimmt und dabei einen Batch von maximal 500 URLs bearbeitet. Du bist also nicht der, welcher am Ende die ganzen Batches zusammenfasst.
</Aufgabe>

<URLs>
- Hier sind die URLs, aus welchen du ein Sample Set auswählen sollst:
{urls_text}
</URLs>

<Zusatzanweisungen>
- Eventuell gibt der Nutzer die ein paar Zusatzanweisungen, um dich etwas mehr zu leiten. Die ursprüngliche Aufgabe bleibt, aber die Zusatzanweisungen können dir helfen, eine Auswahl zu treffen, welche mehr den Vorstellungen des Nutzers entspricht.
{custom_instructions}
</Zusatzanweisungen>"""

INTERESTING_URLS_EXTRACTION_SUMMARIZER_SYSTEM_PROMPT = """<Aufgabe>
- Genauer gesagt, bist du der KI-Agent, welche die ganzen Batches zu einem finalen Sample Set zusammenfasst.
</Aufgabe>

<Batch Sample Sets>
- Hier sind die Batches, welche du zusammenfassen sollst:
{batch_llm_outputs_text}
</Batch Sample Sets>

<Zusatzanweisungen>
- Eventuell gibt der Nutzer die ein paar Zusatzanweisungen, um dich etwas mehr zu leiten. Die ursprüngliche Aufgabe bleibt, aber die Zusatzanweisungen können dir helfen, eine Auswahl zu treffen, welche mehr den Vorstellungen des Nutzers entspricht.
{custom_instructions}
</Zusatzanweisungen>"""


# --- LLM Output models ---


class InterestingUrlsSingleResultlLlmOutput(BaseModel):
    url: str = Field()
    reason: str = Field()


class InterestingUrlsFullResultLlmOutput(BaseModel):
    url_infos: list[InterestingUrlsSingleResultlLlmOutput] = Field()


# --- System prompt methods ---


def generate_interesting_urls_batch_system_prompt(
    urls: list[str], custom_instructions: str
) -> str:
    parts: list[str] = []
    parts.append(GENERAL_SYSTEM_PROMPT)
    parts.append(INTERESTING_URLS_EXTRACTION_COMMON_SYSTEM_PROMPT)

    urls_text = "\n".join([
        f"{x_index + 1}. {x_url}" for x_index, x_url in enumerate(urls)
    ])
    batch_system_prompt = INTERESTING_URLS_EXTRACTION_BATCH_SYSTEM_PROMPT.format(
        urls_text=urls_text, custom_instructions=custom_instructions
    )
    parts.append(batch_system_prompt)

    system_prompt = "\n\n".join(parts)
    return system_prompt


def generate_interesting_urls_summarizer_system_prompt(
    batch_full_result_llm_outputs: list[InterestingUrlsFullResultLlmOutput],
    custom_instructions: str,
) -> str:
    parts: list[str] = []
    parts.append(GENERAL_SYSTEM_PROMPT)
    parts.append(INTERESTING_URLS_EXTRACTION_SUMMARIZER_SYSTEM_PROMPT)

    batch_llm_outputs_text = "\n\n".join([
        f"Batch {x_index + 1}:\n"
        + "\n".join([
            f"- {y_single_result_llm_output.url}\n  Reason: {y_single_result_llm_output.reason}"
            for y_single_result_llm_output in x_batch_full_resultllm_output.url_infos
        ])
        for x_index, x_batch_full_resultllm_output in enumerate(
            batch_full_result_llm_outputs
        )
    ])
    system_prompt = INTERESTING_URLS_EXTRACTION_SUMMARIZER_SYSTEM_PROMPT.format(
        batch_llm_outputs_text=batch_llm_outputs_text,
        custom_instructions=custom_instructions,
    )
    return system_prompt


# --- File helpers ---


def get_interesting_urls_folder_path() -> Path:
    project_folder_path = PROJECT_MANAGER.get_project_folder_path()
    file_path = project_folder_path / "20_interesting_urls"
    file_path.mkdir(exist_ok=True, parents=True)
    return file_path


def get_interesting_urls_results_folder_path() -> Path:
    interesting_urls_folder_path = get_interesting_urls_folder_path()
    results_folder_path = interesting_urls_folder_path / "results"
    results_folder_path.mkdir(exist_ok=True, parents=True)
    return results_folder_path


def save_interesting_urls(urls: list[str]) -> None:
    interesting_urls_results_folder_path = get_interesting_urls_results_folder_path()
    file_path = interesting_urls_results_folder_path / "interesting_urls.txt"
    urls_text = "\n".join(urls)
    with open(file_path, "w") as file:
        file.write(urls_text)


def get_interesting_urls() -> list[str]:
    interesting_urls_results_folder_path = get_interesting_urls_results_folder_path()
    interesting_urls_file_path = (
        interesting_urls_results_folder_path / "interesting_urls.txt"
    )
    with open(interesting_urls_file_path, "r") as file:
        urls = [
            x_line.strip() for x_line in file.readlines() if len(x_line.strip()) > 0
        ]
    return urls


def save_interesting_urls_reason(
    llm_output: InterestingUrlsFullResultLlmOutput,
) -> None:
    interesting_urls_results_folder_path = get_interesting_urls_results_folder_path()
    file_path = interesting_urls_results_folder_path / "interesting_urls_reasons.txt"
    reason = llm_output.model_dump_json(indent=2)
    with open(file_path, "w") as file:
        file.write(reason)


def get_interesting_urls_downloads_folder_path() -> Path:
    interesting_urls_folder_path = get_interesting_urls_folder_path()
    downloaded_folder_path = interesting_urls_folder_path / "downloads"
    downloaded_folder_path.mkdir(exist_ok=True)
    return downloaded_folder_path


def save_downloaded_url(index: int, url: str, html_content: str) -> None:
    downloads_folder_path = get_interesting_urls_downloads_folder_path()
    index_file_name_part = f"{index:03d}"
    url_file_name_part = convert_url_to_file_name(url)
    full_file_name = f"{index_file_name_part}_{url_file_name_part}.html"
    file_path = downloads_folder_path / full_file_name
    with open(file_path) as file:
        file.write(html_content)


# --- Methods ---


async def download_interesting_urls() -> None:
    log_message = "Downloading interesting URLs..."
    logger.info(log_message)

    urls = get_interesting_urls()

    log_message = f"Found {len(urls)} URLs."
    logger.info(log_message)

    async with AsyncWebCrawler() as crawler:
        download_tasks = [crawler.arun(x_url) for x_url in urls]
        results = await asyncio.gather(*download_tasks)

    responses = [x_result.html for x_result in results]

    sub_state_data_packs: list[SubStateDataPack] = []

    for x_url, x_response in zip(urls, responses):
        soup = BeautifulSoup(x_response, "html.parser")
        prettified_html = soup.prettify()

        sub_state_data_pack = SubStateDataPack(
            url=x_url,
            raw_html=prettified_html,
            cleaned_html="",
            raw_markdown="",
            feedback="",
        )
        sub_state_data_packs.append(sub_state_data_pack)

    results_folder_path = get_interesting_urls_results_folder_path()
    downloads_folder_path = get_interesting_urls_downloads_folder_path()
    sub_state = SubState(
        versions_folder_path=downloads_folder_path,
        cloneable_result_folder_path=results_folder_path,
        sub_state_data_packs=sub_state_data_packs,
    )
    SUB_STATE_MANAGER.save_sub_state(sub_state)

    log_message = f"Downloaded {len(urls)} URLs."
    logger.info(log_message)


SUMMARIZER_MAX_BATCHES_PER_LEVEL = 15


async def extract_interesting_urls(custom_instructions: str) -> None:
    log_message = "Extracting interesting URLs..."
    logger.info(log_message)

    non_excluded_urls = get_saved_non_excluded_urls()

    log_message = f"Found {len(non_excluded_urls)} non-excluded URLs."
    logger.info(log_message)

    url_batches = batch_items(non_excluded_urls, INTERESTING_URLS_EXTRACTION_BATCH_SIZE)
    batch_full_result_llm_outputs: list[InterestingUrlsFullResultLlmOutput] = []

    batch_system_prompts = [
        generate_interesting_urls_batch_system_prompt(x_url_batch, custom_instructions)
        for x_url_batch in url_batches
    ]
    batch_full_result_llm_outputs = await call_structured_llm_batch(
        batch_system_prompts,
        InterestingUrlsFullResultLlmOutput,
    )

    log_message = f"Summarizing {len(batch_full_result_llm_outputs)} batches..."
    logger.info(log_message)

    system_prompt = generate_interesting_urls_summarizer_system_prompt(
        batch_full_result_llm_outputs, custom_instructions
    )
    summarized_full_result_llm_output = await call_structured_llm(
        system_prompt,
        InterestingUrlsFullResultLlmOutput,
    )
    urls = [
        x_url_info.url for x_url_info in summarized_full_result_llm_output.url_infos
    ]

    log_message = f"Summarized {len(batch_full_result_llm_outputs)} batches and found {len(urls)} interesting URLs."
    logger.info(log_message)

    save_interesting_urls(urls)
    save_interesting_urls_reason(summarized_full_result_llm_output)

    log_message = f"Extracted {len(urls)} interesting URLs."
    logger.info(log_message)


# =============================================================================
# CSS Selectors
# =============================================================================

# --- Prompts ---

CSS_SELECTORS_EXTRACTION_SYSTEM_PROMPT = """<Prozess>
- Du bist Teil des folgenden Prozesses:
    - Um die Ausschließ-CSS-Selektoren zu bestimmen, wird ein Sample Set an Unterseiten der Webseite analysiert.
    - Dabei werden die heruntergeladenen HTML-Seiten betrachtet und Ausschließ-CSS-Selektoren identifiziert, die "Verschmutzungen" entfernen (Header, Footer, Cookiebanner, Werbung, etc.).
    - Dabei soll alles entfernt werden, was nicht wirklichen Inhalt darstellt. Natürlich sollen Titel, Beschreibungen, Kontaktdaten, Öffnungszeiten, FAQ Akkordions, Info Sidebars etc. alles bleiben. Auch Links, welche zum Inhalt gehören, zu Forms, PDFs, weiteren Informationen sollten beigehalten werden.
    - Aber viel bei einer Webseite ist auch einfach um die eigentlichen Inhalte "drumherum", z. B. Nav, Footer, PopUps, Bedienungshilfen, Socialmedia Widgets, Breadcrumbs, etc. diese sollen alle entfernt werden. Links, welche einfach nur der generellen Seitennavigation angehöhren, z. B. im Footer oder in der Nav sollten ebenfalls entfernt werden.
    - Das Ziel ist es, CSS-Selektoren zu finden, welche auf allen Seiten anwendbar sind, da am Ende das gleiche Set für alle tausende Seiten der Webseite verwendet wird.
    - CSS-Selektoren sollten so generisch wie möglich gehalten werden und z. B. nicht auf bestimmte Titel auf bestimmten Seiten abzielen.
    - Es sollte auch immer das höchstmögliche Element targetiert werden, z. B. sollte natürlich nicht jeder Button einzelnd in einem Cookie-Banner entfernt werden, sondern direkt der ganze Banner oder wenn dieser im Footer ist, welcher auch weg soll, dann direkt der gesamte Footer. So minimieren wir die benötigte Anzahl an CSS-Selektoren.
    - Da mit etwa 20 Sample Seiten gearbeitet wird und das rohe HTML noch sehr lang ist, wird in Batches vorgegangen. Zuerst analysieren mehrere KI-Agenten immer 1 Seite und identifizieren perfekte CSS-Selektoren und begründen auch ihre Auswahl. Es ist wichtig eine gute Begründung zu geben, damit der zusammenfassende KI-Agent die Gedanken hinter den Auswahlen besser versteht. Zusätzlich wird zu jedem CSS-Selektor auch eine Beispiel-Zeilennummer aus dem HTML angegeben. Durch diese wird dann ein Beispiel-Code-Block aus dem HTML geschnitten und auch dem zusammenfassenden KI-Agenten gegeben, damit dieser die Entscheidungen besser nachvollziehen kann und auch beim Kombinieren noch selbst eine gute Entscheidungsgrundlage hat.
    - Damit das HTML an den Agenten nicht zu groß ist werden im Vorhinein schon {precleaning_exclusion_css_selectors_text} Elemente automatisch entfernt.
    - Dann nimmt ein zweiter KI-Agent die Batches und kombiniert diese zu einer finalen Liste an CSS-Selektoren, indem er versucht die CSS-Selektoren so generisch wie möglich zu kombinieren und versucht sicherzustellen, dass die CSS-Selektoren auf allen Seiten anwendbar sind. Dabei gibt es keine Limitierung für die Anzahl an CSS-Selektoren.
    - Diese Selektoren werden dann später in neuracrawl verwendet, um sauberes Markdown zu extrahieren.
</Prozess>"""

CSS_SELECTORS_EXTRACTION_BATCH_SYSTEM_PROMPT = """<Aufgabe>
- Genauer gesagt, bist du der KI-Agent, welcher die CSS-Selektoren für eine einzelne HTML-Seite identifiziert. Du bist also nicht der, welcher am Ende die ganzen Batches zusammenfasst.
</Aufgabe>

<HTML>
- Hier ist die HTML-Seite, welche du analysieren sollst (mit Zeilennummern):
{html_text}
</HTML>

<Zusatzanweisungen>
- Eventuell gibt der Nutzer dir ein paar Zusatzanweisungen, um dich etwas mehr zu leiten. Die ursprüngliche Aufgabe bleibt, aber die Zusatzanweisungen können dir helfen, eine Auswahl zu treffen, welche mehr den Vorstellungen des Nutzers entspricht.
- Manchmal sind diese Anweisungen für dich nicht so relevant, da sie nur auf eine oder nur ein paar bestimmte Seiten abzielen und du für diese nicht verantwortlich bist.
{custom_instructions}
</Zusatzanweisungen>"""

CSS_SELECTORS_EXTRACTION_SUMMARIZER_SYSTEM_PROMPT = """<Aufgabe>
- Genauer gesagt, bist du der KI-Agent, welcher die ganzen Batches zu einer finalen Liste an CSS-Selektoren zusammenfasst.
</Aufgabe>

<Batch CSS Selectors>
- Hier sind die Batches, welche du zusammenfassen sollst:
{batch_llm_outputs_text}
</Batch CSS Selectors>

<Zusatzanweisungen>
- Eventuell gibt der Nutzer dir ein paar Zusatzanweisungen, um dich etwas mehr zu leiten. Die ursprüngliche Aufgabe bleibt, aber die Zusatzanweisungen können dir helfen, eine Auswahl zu treffen, welche mehr den Vorstellungen des Nutzers entspricht.
- Die Batch KI-Agenten sollten diese Anweisungen schon beachtet haben, du kannst es natürlich aber auch nochmal in Betracht ziehen.
{custom_instructions}
</Zusatzanweisungen>"""


# --- LLM Output models ---


class CssSelectorSingleResultLlmOutput(BaseModel):
    css_selector: str = Field()
    reason: str = Field()
    example_line_number: int = Field()


class CssSelectorsFullResultLlmOutput(BaseModel):
    css_selector_infos: list[CssSelectorSingleResultLlmOutput] = Field()


# --- Extended models ---


class ExtendedCssSelectorSingleResult(CssSelectorSingleResultLlmOutput):
    example_html: str = Field()


class ExtendedCssSelectorsFullResult(BaseModel):
    url: str = Field()
    css_selector_infos: list[ExtendedCssSelectorSingleResult] = Field()


# --- System prompt methods ---


def generate_css_selectors_batch_system_prompt(
    html_content: str, custom_instructions: str
) -> str:
    parts: list[str] = []
    parts.append(GENERAL_SYSTEM_PROMPT)

    precleaning_exclusion_css_selectors_text = ", ".join(
        CLEANER.PRECLEANING_EXCLUSION_CSS_SELECTORS
    )
    css_selectors_extraction_system_prompt = CSS_SELECTORS_EXTRACTION_SYSTEM_PROMPT.format(
        precleaning_exclusion_css_selectors_text=precleaning_exclusion_css_selectors_text
    )
    parts.append(css_selectors_extraction_system_prompt)

    html_lines = html_content.split("\n")
    numbered_html_lines = [
        f"{x_index + 1:5d} | {x_line}" for x_index, x_line in enumerate(html_lines)
    ]
    html_text = "\n".join(numbered_html_lines)
    batch_system_prompt = CSS_SELECTORS_EXTRACTION_BATCH_SYSTEM_PROMPT.format(
        html_text=html_text, custom_instructions=custom_instructions
    )
    parts.append(batch_system_prompt)

    system_prompt = "\n\n".join(parts)
    return system_prompt


def generate_css_selectors_summarizer_system_prompt(
    batch_llm_outputs: list[ExtendedCssSelectorsFullResult],
    custom_instructions: str,
) -> str:
    parts: list[str] = []
    parts.append(GENERAL_SYSTEM_PROMPT)
    parts.append(CSS_SELECTORS_EXTRACTION_SYSTEM_PROMPT)

    batch_llm_outputs_text = "\n\n".join([
        f"Batch {x_index + 1} (URL: {x_batch_llm_output.url}):\n"
        + "\n".join([
            f"- CSS Selector: {y_selector_output.css_selector}\n  Reason: {y_selector_output.reason}\n  Example HTML:\n{y_selector_output.example_html}"
            for y_selector_output in x_batch_llm_output.css_selector_infos
        ])
        for x_index, x_batch_llm_output in enumerate(batch_llm_outputs)
    ])
    system_prompt = CSS_SELECTORS_EXTRACTION_SUMMARIZER_SYSTEM_PROMPT.format(
        batch_llm_outputs_text=batch_llm_outputs_text,
        custom_instructions=custom_instructions,
    )
    parts.append(system_prompt)

    system_prompt = "\n\n".join(parts)
    return system_prompt


# --- File helpers ---


def get_css_selectors_folder_path() -> Path:
    project_folder_path = PROJECT_MANAGER.get_project_folder_path()
    file_path = project_folder_path / "30_css_selectors"
    file_path.mkdir(exist_ok=True, parents=True)
    return file_path


def get_css_selectors_results_folder_path() -> Path:
    css_selectors_folder_path = get_css_selectors_folder_path()
    results_folder_path = css_selectors_folder_path / "results"
    results_folder_path.mkdir(exist_ok=True, parents=True)
    return results_folder_path


def get_css_selectors_file_path() -> Path:
    css_selectors_results_folder_path = get_css_selectors_results_folder_path()
    file_path = css_selectors_results_folder_path / "css_selectors.txt"
    return file_path


def save_css_selectors(css_selectors: list[str]) -> None:
    css_selectors_results_folder_path = get_css_selectors_results_folder_path()
    file_path = css_selectors_results_folder_path / "css_selectors.txt"
    selectors_text = "\n".join(css_selectors)
    with open(file_path, "w") as file:
        file.write(selectors_text)


def get_saved_css_selectors() -> list[str]:
    file_path = get_css_selectors_file_path()
    if not file_path.exists():
        return []
    with open(file_path, "r") as file:
        selectors = [
            x_line.strip() for x_line in file.readlines() if len(x_line.strip()) > 0
        ]
    return selectors


def save_css_selectors_reason(
    llm_output: CssSelectorsFullResultLlmOutput,
) -> None:
    css_selectors_results_folder_path = get_css_selectors_results_folder_path()
    file_path = css_selectors_results_folder_path / "css_selectors_reasons.txt"
    reason = llm_output.model_dump_json(indent=2)
    with open(file_path, "w") as file:
        file.write(reason)


def get_css_selectors_downloads_folder_path() -> Path:
    css_selectors_folder_path = get_css_selectors_folder_path()
    downloads_folder_path = css_selectors_folder_path / "downloads"
    downloads_folder_path.mkdir(exist_ok=True)
    return downloads_folder_path


# --- Utilities ---


def extract_example_html_lines(html_content: str, line_number: int) -> str:
    html_lines = html_content.split("\n")
    start_index = max(0, line_number - 4)
    end_index = min(len(html_lines), line_number + 3)

    example_lines = html_lines[start_index:end_index]
    numbered_example_lines = [
        f"{start_index + x_index + 1:5d} | {x_line}"
        for x_index, x_line in enumerate(example_lines)
    ]
    return "\n".join(numbered_example_lines)


def hydrate_full_results_to_extended_full_results(
    full_results: list[CssSelectorsFullResultLlmOutput],
    sub_state_data_packs: list[SubStateDataPack],
) -> list[ExtendedCssSelectorsFullResult]:
    extended_full_results: list[ExtendedCssSelectorsFullResult] = []

    for x_sub_state_data_pack, x_llm_output in zip(sub_state_data_packs, full_results):
        extended_single_results: list[ExtendedCssSelectorSingleResult] = []

        for x_css_selector_info in x_llm_output.css_selector_infos:
            example_html = extract_example_html_lines(
                x_sub_state_data_pack.cleaned_html,
                x_css_selector_info.example_line_number,
            )
            extended_single_result = ExtendedCssSelectorSingleResult(
                css_selector=x_css_selector_info.css_selector,
                reason=x_css_selector_info.reason,
                example_line_number=x_css_selector_info.example_line_number,
                example_html=example_html,
            )
            extended_single_results.append(extended_single_result)

        extended_full_result = ExtendedCssSelectorsFullResult(
            url=x_sub_state_data_pack.url,
            css_selector_infos=extended_single_results,
        )
        extended_full_results.append(extended_full_result)

    return extended_full_results


# --- Methods ---


async def _hierarchical_summarize_css_selectors(
    batch_results: list[ExtendedCssSelectorsFullResult],
    custom_instructions: str,
    level: int = 1,
) -> CssSelectorsFullResultLlmOutput:
    """Recursively summarize CSS selector batch results in chunks to avoid context window limits."""
    if len(batch_results) <= SUMMARIZER_MAX_BATCHES_PER_LEVEL:
        log_message = f"Summarizing {len(batch_results)} batches (level {level})..."
        logger.info(log_message)

        system_prompt = generate_css_selectors_summarizer_system_prompt(
            batch_results, custom_instructions
        )
        return await call_structured_llm(
            system_prompt,
            CssSelectorsFullResultLlmOutput,
        )

    # Split into chunks and summarize each
    chunks = batch_items(batch_results, SUMMARIZER_MAX_BATCHES_PER_LEVEL)
    log_message = f"Too many batches ({len(batch_results)}), splitting into {len(chunks)} chunks for level {level} summarization..."
    logger.info(log_message)

    chunk_summaries: list[ExtendedCssSelectorsFullResult] = []
    for chunk_idx, chunk in enumerate(chunks):
        log_message = f"Summarizing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} batches) at level {level}..."
        logger.info(log_message)

        system_prompt = generate_css_selectors_summarizer_system_prompt(
            chunk, custom_instructions
        )
        chunk_summary = await call_structured_llm(
            system_prompt,
            CssSelectorsFullResultLlmOutput,
        )
        # Convert back to ExtendedCssSelectorsFullResult for next level
        extended_chunk_summary = ExtendedCssSelectorsFullResult(
            url=f"Level {level} Chunk {chunk_idx + 1} Summary",
            css_selector_infos=[
                ExtendedCssSelectorSingleResult(
                    css_selector=x.css_selector,
                    reason=x.reason,
                    example_line_number=x.example_line_number,
                    example_html="(from previous summarization)",
                )
                for x in chunk_summary.css_selector_infos
            ],
        )
        chunk_summaries.append(extended_chunk_summary)

    # Recursively summarize the chunk summaries
    return await _hierarchical_summarize_css_selectors(
        chunk_summaries, custom_instructions, level + 1
    )


async def extract_css_selectors(
    custom_instructions: str, interesting_urls_save_state_version: str | None = None
) -> None:
    log_message = "Extracting CSS selectors..."
    logger.info(log_message)

    interesting_urls_downloads_folder_path = (
        get_interesting_urls_downloads_folder_path()
    )
    interesting_urls_sub_state = SUB_STATE_MANAGER.get_sub_state(
        interesting_urls_downloads_folder_path, interesting_urls_save_state_version
    )
    sub_state_data_packs = interesting_urls_sub_state.sub_state_data_packs

    log_message = f"Found {len(sub_state_data_packs)} pages."
    logger.info(log_message)

    log_message = f"Pre-cleaning HTML for {len(sub_state_data_packs)} pages..."
    logger.info(log_message)

    precleaned_sub_state_data_packs: list[SubStateDataPack] = []

    for x_sub_state_data_pack in sub_state_data_packs:
        precleaned_html = CLEANER.preclean_html(x_sub_state_data_pack.raw_html)
        precleaned_sub_state_data_pack = x_sub_state_data_pack.model_copy()
        precleaned_sub_state_data_pack.cleaned_html = precleaned_html
        precleaned_sub_state_data_packs.append(precleaned_sub_state_data_pack)

    log_message = f"Pre-cleaned HTML for {len(precleaned_sub_state_data_packs)} pages."
    logger.info(log_message)

    batch_system_prompts = [
        generate_css_selectors_batch_system_prompt(
            x_sub_state_data_pack.cleaned_html, custom_instructions
        )
        for x_sub_state_data_pack in precleaned_sub_state_data_packs
    ]
    for x_prompt in batch_system_prompts:
        print(len(x_prompt))
    batch_full_results = await call_structured_llm_batch(
        batch_system_prompts,
        CssSelectorsFullResultLlmOutput,
    )
    extended_batch_full_results = hydrate_full_results_to_extended_full_results(
        batch_full_results, precleaned_sub_state_data_packs
    )

    # Hierarchical summarization to handle many batches
    summarized_full_result = await _hierarchical_summarize_css_selectors(
        extended_batch_full_results, custom_instructions
    )
    css_selectors = [
        x_selector_output.css_selector
        for x_selector_output in summarized_full_result.css_selector_infos
    ]

    log_message = (
        f"Found {len(css_selectors)} CSS selectors after hierarchical summarization."
    )
    logger.info(log_message)

    save_css_selectors(css_selectors)
    save_css_selectors_reason(summarized_full_result)

    log_message = f"Processing {len(sub_state_data_packs)} pages..."
    logger.info(log_message)

    processed_sub_state_data_packs: list[SubStateDataPack] = []
    for x_sub_state_data_pack in sub_state_data_packs:
        cleaned_html = CLEANER.clean_html(x_sub_state_data_pack.raw_html, css_selectors)
        raw_markdown = CLEANER.convert_html_to_markdown(cleaned_html)

        processed_sub_state_data_pack = x_sub_state_data_pack.model_copy()
        processed_sub_state_data_pack.cleaned_html = cleaned_html
        processed_sub_state_data_pack.raw_markdown = raw_markdown
        processed_sub_state_data_packs.append(processed_sub_state_data_pack)

    log_message = f"Processed {len(processed_sub_state_data_packs)} pages."
    logger.info(log_message)

    downloads_folder_path = get_css_selectors_downloads_folder_path()
    results_folder_path = get_css_selectors_results_folder_path()
    sub_state = SubState(
        versions_folder_path=downloads_folder_path,
        cloneable_result_folder_path=results_folder_path,
        sub_state_data_packs=processed_sub_state_data_packs,
    )
    SUB_STATE_MANAGER.save_sub_state(sub_state)

    log_message = f"Extracted {len(css_selectors)} CSS selectors."
    logger.info(log_message)


def apply_css_selectors(
    interesting_urls_save_state_version: str | None = None,
    css_selectors: list[str] | None = None,
) -> None:
    """
    Apply CSS selectors to pages without LLM calls.

    Reads CSS selectors from css_selectors.txt (or uses provided list) and applies them
    to all pages, saving a new version.

    Args:
        interesting_urls_save_state_version: Version of interesting URLs to use (None = latest)
        css_selectors: CSS selectors to apply (None = read from css_selectors.txt)
    """
    log_message = "Applying CSS selectors..."
    logger.info(log_message)

    # Load CSS selectors
    if css_selectors is None:
        css_selectors = get_saved_css_selectors()

    if len(css_selectors) == 0:
        raise ValueError(
            "No CSS selectors found. Run extract_css_selectors() first or provide css_selectors."
        )

    log_message = f"Loaded {len(css_selectors)} CSS selectors."
    logger.info(log_message)

    # Load pages from interesting URLs
    interesting_urls_downloads_folder_path = (
        get_interesting_urls_downloads_folder_path()
    )
    interesting_urls_sub_state = SUB_STATE_MANAGER.get_sub_state(
        interesting_urls_downloads_folder_path, interesting_urls_save_state_version
    )
    sub_state_data_packs = interesting_urls_sub_state.sub_state_data_packs

    log_message = f"Processing {len(sub_state_data_packs)} pages..."
    logger.info(log_message)

    processed_sub_state_data_packs: list[SubStateDataPack] = []
    for x_sub_state_data_pack in sub_state_data_packs:
        cleaned_html = CLEANER.clean_html(x_sub_state_data_pack.raw_html, css_selectors)
        raw_markdown = CLEANER.convert_html_to_markdown(cleaned_html)

        processed_sub_state_data_pack = x_sub_state_data_pack.model_copy()
        processed_sub_state_data_pack.cleaned_html = cleaned_html
        processed_sub_state_data_pack.raw_markdown = raw_markdown
        processed_sub_state_data_packs.append(processed_sub_state_data_pack)

    log_message = f"Processed {len(processed_sub_state_data_packs)} pages."
    logger.info(log_message)

    downloads_folder_path = get_css_selectors_downloads_folder_path()
    results_folder_path = get_css_selectors_results_folder_path()
    sub_state = SubState(
        versions_folder_path=downloads_folder_path,
        cloneable_result_folder_path=results_folder_path,
        sub_state_data_packs=processed_sub_state_data_packs,
    )
    SUB_STATE_MANAGER.save_sub_state(sub_state)

    next_version = SUB_STATE_MANAGER.get_latest_version(downloads_folder_path)
    log_message = f"Applied {len(css_selectors)} CSS selectors to {len(processed_sub_state_data_packs)} pages. Saved as {next_version}."
    logger.info(log_message)


# =============================================================================
# Public API - Convenience exports
# =============================================================================

__all__ = [
    # Project management
    "PROJECT_MANAGER",
    "create_sitemap_urls_file",
    # Sitemap
    "extract_sitemap_urls",
    "extract_frequent_sitemap_urls",
    "extract_url_extensions",
    # URL regexes
    "extract_url_regexes",
    # Interesting URLs
    "extract_interesting_urls",
    "download_interesting_urls",
    # CSS selectors
    "extract_css_selectors",
    "apply_css_selectors",
]
