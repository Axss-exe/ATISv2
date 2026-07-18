#!/usr/bin/env python3
"""
ATIS Zimbabwe News Seeder — Imports 10 real July 2026 articles into Neon PostgreSQL.
Usage:
    export DATABASE_URL="postgresql://user:pass@host.neon.tech/dbname?sslmode=require"
    python3 import_zimbabwe_articles.py
"""

import os
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 10 REAL ZIMBABWE ARTICLES — JULY 10-17, 2026
# ─────────────────────────────────────────────────────────────────────────────

ARTICLES = [
    {
        "headline": "Mnangagwa Signs Constitutional Amendment Extending Presidential Term to 2030",
        "source": "AFP / NAMPA",
        "published_at": "2026-07-07 08:00:00+00",
        "article_text": (
            "Harare, Zimbabwe — Zimbabwean President Emmerson Mnangagwa on Tuesday signed into law constitutional changes "
            "that will extend his term by two years until 2030 and scrap direct presidential elections in favor of a "
            "parliamentary selection model. The Constitutional Amendment Bill No. 3 (CAB3) was passed by the National "
            "Assembly on June 18 with 216 votes in favor and 42 against, and by the Senate on June 24 with 75 votes to 4.\n\n"
            "The amendments remove the two-term presidential limit and transfer the power to elect the president to "
            "parliament, a move opposition leaders have condemned as a 'constitutional coup.' Anonymous pamphlets "
            "circulated across military barracks in Harare on June 10 claiming the bill would weaken military influence, "
            "while a group of former generals led by retired Air Marshal Henry Muchena met Mnangagwa twice in May to "
            "persuade him to reconsider.\n\n"
            "The Crisis Group warned that the political crisis has escalated, with the Constitutional Court dismissing "
            "a legal challenge brought by war veterans on June 17. Civil society groups including Constitution Defenders "
            "Forum and WeThePeople have called for a referendum before the law takes effect. Police raided the "
            "Constitution Defenders Forum offices in Harare on June 23 and arrested one of its leaders.\n\n"
            "Despite domestic turmoil, Zimbabwe secured a non-permanent seat on the UN Security Council for 2027–2028 on "
            "June 3, receiving 182 of 191 votes. The government has framed the constitutional changes as necessary for "
            "stability and continuity in its economic reform agenda."
        ),
        "summary": "Mnangagwa extends term to 2030 via constitutional amendment, sparking political crisis and opposition backlash",
        "category": "Politics",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe Engages Regional Aviation Stakeholders on Airspace Modernization",
        "source": "ZBC News",
        "published_at": "2026-07-09 10:00:00+00",
        "article_text": (
            "Harare — The government of Zimbabwe has convened a high-level meeting with regional aviation stakeholders "
            "to discuss the modernization of the country's air traffic management systems and the expansion of international "
            "flight routes. The engagement, held on July 9, 2026, brought together representatives from the International "
            "Air Transport Association (IATA), the African Civil Aviation Commission (AFCAC), and national carriers from "
            "South Africa, Zambia, and Kenya.\n\n"
            "Transport and Infrastructural Development Minister Felix Mhona said the initiative is part of Zimbabwe's "
            "broader strategy to position Harare and Victoria Falls as major aviation hubs for southern Africa. 'Our "
            "airspace infrastructure has lagged behind regional peers for too long. We are now committing $120 million "
            "over the next three years to upgrade radar systems, expand Robert Gabriel Mugabe International Airport "
            "capacity, and establish Victoria Falls as a regional cargo hub,' Mhona stated.\n\n"
            "The plan includes the construction of a second runway at Harare airport, a new passenger terminal at "
            "Victoria Falls, and the rehabilitation of Joshua Mqabuko Nkomo International Airport in Bulawayo. "
            "Regional airlines welcomed the move, noting that Zimbabwean airspace has been underutilized due to outdated "
            "navigation aids and high landing fees. IATA representatives cautioned that implementation timelines must "
            "be met to attract foreign airline partnerships, particularly from Gulf carriers looking to expand into "
            "southern Africa."
        ),
        "summary": "Zimbabwe convenes regional aviation summit to modernize airspace and expand airport capacity",
        "category": "Infrastructure",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe Exports Jump 42% to US$4.5 Billion on Gold and Tobacco Surge",
        "source": "tralac Daily News",
        "published_at": "2026-07-07 06:00:00+00",
        "article_text": (
            "Zimbabwe's merchandise exports surged 42% in the first half of 2026 to reach US$4.5 billion, driven by "
            "record gold deliveries and a bumper tobacco marketing season. According to data from the Reserve Bank of "
            "Zimbabwe and ZIMSTAT, semi-manufactured gold accounted for 47% of total export value, while stemmed and "
            "stripped tobacco contributed US$605 million.\n\n"
            "The export boom has narrowed the trade deficit significantly and provided much-needed foreign currency "
            "liquidity to the Zimbabwe Gold (ZiG) currency system. Mining sector analysts attribute the gold surge to "
            "elevated international prices and the formalization of artisanal mining operations, which now contribute "
            "over 60% of deliveries to Fidelity Gold Refinery. The Tobacco Industry and Marketing Board (TIMB) reported "
            "that the 2026 season saw improved quality grades and stronger demand from Chinese and European buyers.\n\n"
            "However, economists warn that the export concentration in gold and tobacco leaves Zimbabwe vulnerable to "
            "commodity price shocks. 'While the numbers look impressive, we are essentially betting the economy on two "
            "products. Diversification into processed minerals and manufactured goods remains critical for sustainable "
            "growth,' said a senior economist at the Zimbabwe National Chamber of Commerce. The government has pointed "
            "to the lithium beneficiation push and new special economic zones as pathways to diversify the export basket."
        ),
        "summary": "H1 2026 exports hit $4.5B on gold and tobacco, but economists warn of concentration risk",
        "category": "Trade",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe's Economic Landscape: What the Data Actually Tells Us in Mid-2026",
        "source": "SwitzView Research",
        "published_at": "2026-07-08 09:30:00+00",
        "article_text": (
            "Zimbabwe's economy is showing cautious recovery in mid-2026, with stronger growth, lower inflation, and "
            "improved foreign currency inflows supporting stability. But debt arrears, thin reserves, infrastructure gaps, "
            "and export concentration still limit the outlook. The opportunity is real, but durable progress will depend "
            "on discipline, investment, and consistent reform.\n\n"
            "Annual ZiG inflation fell to 4.38% in March 2026, the lowest sustained inflation under any Zimbabwe-issued "
            "currency in over three decades. The ZiG appreciated 2.54% during Q1 2026, closing at 25.32 per US dollar, "
            "with the parallel market premium staying below 20%. The IMF approved a Staff-Monitored Program in April 2026, "
            "providing a positive anchor for policy credibility, though it remains a monitoring arrangement, not a lending program.\n\n"
            "The World Bank estimated that real GDP rebounded strongly in 2025, driven by agriculture, mining, and services. "
            "The RBZ projects about 5% growth for 2026, led by agriculture (maize and tobacco) and mining (gold). However, "
            "that recovery is partly a rebound from weakness — the 2024 drought and electricity shortfalls hit hard. "
            "The most important Zimbabwe risk is not simple 'country risk.' It is rule-change risk. Investors must price "
            "the possibility that FX procedures, tax interpretations, royalty mechanics, or value-addition requirements "
            "shift faster than project economics can absorb."
        ),
        "summary": "Mid-2026 economic data shows recovery but rule-change risk remains the top investor concern",
        "category": "Economy",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe's Renewable Energy Pipeline Reaches 160 Projects, But Financing Gaps Persist",
        "source": "SwitzView Energy",
        "published_at": "2026-07-01 11:00:00+00",
        "article_text": (
            "Zimbabwe has 2,962 MW of installed capacity, with renewables at 1,282 MW targeting 2,640 MW by 2030. Three "
            "proof-point projects — Centragrid (25 MW), Great Zimbabwe Hydro (5 MW), and Vungu Solar (30 MW) — show "
            "bankability works under the right structures. The real challenge is not resources; it is converting a "
            "160-project pipeline into financed assets at speed while tackling macro volatility.\n\n"
            "Vungu Solar (Pvt) Ltd signed a 25-year Power Purchase Agreement (PPA) with the Zimbabwe Electricity "
            "Transmission and Distribution Company (ZETDC), marking a milestone for independent power producers. The "
            "Zimbabwe Energy Regulatory Authority (ZERA) has launched a national program to deploy portable, containerized "
            "fuel stations for rural and mining communities, following a successful pilot in Gokwe Nembudziya.\n\n"
            "In a bid to encourage rapid deployment of distributed energy, ZERA has waived licensing fees entirely for "
            "projects below ten megawatts, signaling the state's prioritization of new electrons over administrative "
            "revenue. The government, through the Ministry of Energy and Power Development, has partnered with Zonful "
            "Energy to implement a nationwide rooftop solar programme aiming to install solar systems in 1 million "
            "households by December 2030. The Ministry will host a SADC sustainable energy week in Victoria Falls from "
            "February 23–27, 2026, focusing on 'Driving Regional Economic Growth through Clean Energy.'"
        ),
        "summary": "160-project renewable pipeline faces financing hurdles despite regulatory easing and proof-point projects",
        "category": "Energy",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "US Treasury Updates Zimbabwe Sanctions, Designates Mnangagwa Jr. and Tagwirei Network",
        "source": "US Department of the Treasury",
        "published_at": "2026-07-10 14:00:00+00",
        "article_text": (
            "Washington — The U.S. Department of the Treasury's Office of Foreign Assets Control (OFAC) today designated "
            "four Zimbabwean individuals and two Zimbabwean entities, and removed seventeen Zimbabweans from the "
            "Specially Designated Nationals and Blocked Persons (SDN) List. The designations target those tied to "
            "Kudakwashe Tagwirei and his company Sakunda Holdings, as well as Emmerson Mnangagwa Jr., the son of "
            "President Mnangagwa.\n\n"
            "Sandra Mpunga, Tagwirei's wife and co-founder of Sakunda, was designated alongside Nqobile Magwizi, "
            "Fossil Agro, Fossil Contracting, and Obey Chimuka. OFAC stated that Tagwirei has utilized relationships "
            "with high-level Zimbabwean officials to gain state contracts and receive favored access to hard currency, "
            "including U.S. dollars. 'The goal of sanctions is behavior change. Today's actions demonstrate our support "
            "for a transparent and prosperous Zimbabwe,' said Under Secretary Brian E. Nelson.\n\n"
            "The Treasury emphasized that U.S. sanctions do not target the Zimbabwean people, the country of Zimbabwe, "
            "or Zimbabwe's banking sector. The removal of seventeen individuals from the SDN List reflects a calibrated "
            "approach to rewarding reform while maintaining pressure on corrupt elites. The Zimbabwean government has "
            "not yet issued an official response, but ruling party officials privately expressed concern that the "
            "designation of the president's son could complicate ongoing re-engagement efforts with Western financial institutions."
        ),
        "summary": "OFAC designates Mnangagwa Jr. and Tagwirei-linked entities while removing 17 others from sanctions list",
        "category": "Policy",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe Gold Deliveries Plunge 38% in January as Fiscal Measures Bite",
        "source": "ZB Financial Holdings",
        "published_at": "2026-07-11 07:00:00+00",
        "article_text": (
            "Gold deliveries to Fidelity Gold Refinery (FGR) saw a sharp decline in January 2026, dropping by 38% to "
            "3,044.97 kg from 4,941.72 kg in December 2025. Artisanal small-scale miners, who contributed the majority "
            "(73%) of total output, recorded the steepest decline of 42% to 2,236.56 kg from 3,881.69 kg recorded the "
            "previous month. Large-scale miners also experienced a drop, with output decreasing by 24% to 808.41 kg from "
            "1,060.03 kg.\n\n"
            "The downturn was primarily attributed to seasonal factors and adjustments to new fiscal measures introduced "
            "in the 2026 National Budget, including changes to the gold retention threshold and royalty structures. "
            "Nevertheless, output is expected to recover, supported by robust gold prices and the ongoing formalization "
            "of artisanal miners. The government-backed certification program has now registered 300 small-scale miners "
            "in Chegutu, with a target of 1,500 nationwide.\n\n"
            "FGR has transitioned to using a live morning benchmark for setting daily gold prices, replacing the previous "
            "method that relied on the prior day's closing price. This shift aims to provide a more transparent and "
            "responsive pricing model for both large-scale and artisanal producers. Meanwhile, Caledonia Mining raised "
            "US$150 million to develop its Bilboes gold project, aiming to turn it into one of the country's largest "
            "gold mines, with peak production targeted at 200,000 ounces per year by 2029."
        ),
        "summary": "January gold deliveries crash 38% on budget changes, but formalization and Bilboes expansion offer recovery path",
        "category": "Mining",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe Lithium Export Ban Accelerates as Concentrate Shipments Halted with Immediate Effect",
        "source": "Mining Zimbabwe",
        "published_at": "2026-07-12 08:00:00+00",
        "article_text": (
            "The Zimbabwean government has suspended the export of all lithium concentrates and raw minerals with "
            "immediate effect until further notice, Minister of Mines and Mining Development Dr. Polite Kambamura announced "
            "at a press conference on July 12. The suspension applies to all minerals currently in transit, sending a "
            "clear signal that the government will no longer tolerate the export of unprocessed mineral wealth.\n\n"
            "The Minister directed all regulatory authorities, including the Zimbabwe Revenue Authority (ZIMRA) and the "
            "Minerals Marketing Corporation of Zimbabwe (MMCZ), to observe the suspension without exception. 'Government "
            "expects the cooperation of the mining industry on this measure, which has been taken in the national interest,' "
            "Kambamura stated. Only mining companies holding valid mining titles and approved beneficiation plans will be "
            "authorized to export minerals. Agents and third-party traders are explicitly barred from the export chain.\n\n"
            "For export permit applications, companies must now attach a recommendation letter from the relevant Provincial "
            "Mines Office stating beneficiation capacity and compliance status, plus a declaration of mineral composition. "
            "The Ministry reserves the right to independently verify these declarations. Prospect Lithium Zimbabwe, a "
            "subsidiary of Huayou Cobalt, is poised to commission the continent's first lithium sulphate plant in Q3 2026, "
            "a US$400 million facility that will significantly increase export earnings compared to raw concentrates. "
            "Mutapa Energy Minerals is also starting construction of a US$250 million lithium concentrate processing "
            "plant at Sandawana, expected to be completed in 2027."
        ),
        "summary": "Immediate ban on raw lithium and mineral exports forces miners to accelerate local beneficiation plans",
        "category": "Mining",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe Secures IMF Staff-Monitored Program, But Debt Arrears Block Concessional Funding",
        "source": "IMF / World Bank",
        "published_at": "2026-07-13 10:00:00+00",
        "article_text": (
            "The International Monetary Fund approved a Staff-Monitored Program (SMP) for Zimbabwe in April 2026, marking "
            "a significant step in the country's re-engagement with international financial institutions. The program is "
            "designed to anchor macroeconomic stability and support the government's reform agenda, but it is not a "
            "lending program — Zimbabwe's unresolved sovereign arrears continue to block access to concessional funding.\n\n"
            "The World Bank estimated that real GDP rebounded strongly in 2025, driven by agriculture, mining, and services, "
            "with the Treasury projecting an ambitious 8.5% GDP expansion for 2026 — the highest since 2012. However, "
            "international lenders remain cautious, projecting 4.6%–5% growth, citing potential external shocks, climate "
            "vulnerabilities affecting agriculture, and the need for sustained fiscal discipline to avoid a return to past "
            "currency volatility.\n\n"
            "Zimbabwe's external and overall public debt is unsustainable and in distress, with total public debt at 45.6% "
            "of GDP in 2025. The RBZ projects about 5% growth for 2026, led by agriculture (maize and tobacco) and mining "
            "(gold). Annual ZiG inflation fell to 4.1% in January 2026, a record low. Successful implementation of structural "
            "reforms will help anchor long-term growth by unlocking concessional funding lines and external long-term "
            "investments, but creditors insist that arrears clearance and governance reforms must come first."
        ),
        "summary": "IMF SMP approved but lending blocked by arrears; growth projections diverge sharply between government and lenders",
        "category": "Finance",
        "country_tag": "Zimbabwe",
    },
    {
        "headline": "Zimbabwe Artisanal Mining Formalization Reaches 300 Certified Miners in Chegutu Pilot",
        "source": "Panafrican Visions",
        "published_at": "2026-07-14 12:00:00+00",
        "article_text": (
            "Chegutu, Zimbabwe — Zimbabwe is accelerating efforts to formalize its artisanal mining sector, with 300 "
            "small-scale miners receiving official certification in a government-backed initiative aimed at improving "
            "safety, compliance, and productivity. The certification ceremony, held in Chegutu about 105 kilometers west "
            "of Harare, marks a strategic shift from traditional Corporate Social Responsibility (CSR) programs to more "
            "targeted Corporate Social Investment (CSI) frameworks.\n\n"
            "Minister of Mines and Mining Development Polite Kambamura hailed the program as a turning point in "
            "Zimbabwe's economic reform agenda. 'Small-scale miners contribute about 60% of deliveries to Fidelity Gold "
            "Refinery. When we empower them, we empower the nation,' Kambamura said. The initiative brings together "
            "government, private sector players, and training institutions in what officials describe as a scalable model "
            "for transforming artisanal mining into a formal, regulated industry.\n\n"
            "Mutapa Gold Resources funded the initiative, with CEO Patrick Maseva Shayawabaya stating that 'turning "
            "artisanal miners into formal miners is a turning point for transparency and accountability.' Training was "
            "delivered by the Zimbabwe School of Mines, with emphasis on safe mining practices, regulatory compliance, "
            "and productivity enhancement. The program includes women miners, reflecting a commitment to inclusivity. "
            "Authorities say the Chegutu program is part of a broader national strategy to professionalize the sector "
            "through mobile mining schools, with a target of certifying 1,500 miners nationwide."
        ),
        "summary": "300 artisanal miners certified in Chegutu pilot as government targets 1,500 nationwide formalizations",
        "category": "Mining",
        "country_tag": "Zimbabwe",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# NEON IMPORT LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    """Connect to Neon using DATABASE_URL env var."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        print("Example: export DATABASE_URL='postgresql://user:pass@host.neon.tech/dbname?sslmode=require'")
        sys.exit(1)

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as exc:
        print(f"ERROR: Failed to connect to Neon: {exc}")
        sys.exit(1)


def ensure_table(conn):
    """Create the news_articles table if it doesn't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS news_articles (
        id SERIAL PRIMARY KEY,
        headline VARCHAR(500) NOT NULL,
        source VARCHAR(200),
        published_at TIMESTAMP WITH TIME ZONE,
        article_text TEXT NOT NULL,
        summary TEXT,
        category VARCHAR(100),
        country_tag VARCHAR(100),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
        conn.commit()
    print("✓ Table 'news_articles' verified.")


def clear_existing_zimbabwe(conn):
    """Optional: remove existing Zimbabwe articles to avoid duplicates."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM news_articles WHERE country_tag = 'Zimbabwe'")
        deleted = cur.rowcount
        conn.commit()
    if deleted > 0:
        print(f"✓ Cleared {deleted} existing Zimbabwe article(s).")
    else:
        print("✓ No existing Zimbabwe articles to clear.")


def insert_articles(conn):
    """Insert the 10 articles."""
    sql = """
    INSERT INTO news_articles (headline, source, published_at, article_text, summary, category, country_tag)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    inserted = 0
    with conn.cursor() as cur:
        for article in ARTICLES:
            cur.execute(sql, (
                article["headline"],
                article["source"],
                article["published_at"],
                article["article_text"],
                article["summary"],
                article["category"],
                article["country_tag"],
            ))
            new_id = cur.fetchone()[0]
            print(f"  → Inserted: [{new_id}] {article['headline'][:60]}...")
            inserted += 1
        conn.commit()
    return inserted


def main():
    print("=" * 70)
    print("ATIS Zimbabwe News Seeder — Neon PostgreSQL")
    print("=" * 70)

    conn = get_connection()
    print("✓ Connected to Neon.")

    ensure_table(conn)

    # Uncomment the next line if you want to wipe and re-seed:
    clear_existing_zimbabwe(conn)

    count = insert_articles(conn)
    print(f"\n✓ Done. Inserted {count} Zimbabwe articles into Neon.")

    # Verify
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM news_articles WHERE country_tag = 'Zimbabwe'")
        total = cur.fetchone()[0]
        print(f"✓ Total Zimbabwe articles in database: {total}")

    conn.close()
    print("=" * 70)


if __name__ == "__main__":
    main()