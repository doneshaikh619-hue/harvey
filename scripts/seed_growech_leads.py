"""Seed Growech Solution Target Customers into Harvey's SQLite database."""

import asyncio
import json
import logging
from pathlib import Path

from harvey.models.company import Company
from harvey.models.prospect import Prospect
from harvey.state import StateManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("seed_growech")


async def main():
    root = Path(__file__).resolve().parent.parent
    seed_file = root / "campaigns" / "growech_customers_seed.json"
    if not seed_file.exists():
        seed_file = root / "data" / "growech_customers_seed.json"

    if not seed_file.exists():
        logger.error(f"Seed file not found at {seed_file}")
        return

    with open(seed_file, "r", encoding="utf-8") as f:
        leads = json.load(f)

    state = StateManager()
    await state.init_db()

    logger.info(f"Loaded {len(leads)} pre-configured Growech target customers.")

    for lead in leads:
        company_name = lead["company_name"]
        domain = lead["email"].split("@")[-1]

        # 1. Ensure company exists
        company = await state.get_company_by_domain(domain)
        if not company:
            company = Company(
                name=company_name,
                domain=domain,
                industry=lead.get("industry", ""),
                location=lead.get("location", ""),
            )
            company_id = await state.add_company(company)
            logger.info(f"Added company: {company_name} ({domain}) -> ID: {company_id}")
        else:
            company_id = company.id

        # 2. Add prospect
        prospect = Prospect(
            company_id=company_id,
            email=lead["email"],
            first_name=lead["first_name"],
            last_name=" ".join(lead["contact_name"].split()[1:]),
            title=lead["title"],
            fit_score=lead.get("score", 90),
            notes=lead.get("notes", ""),
        )

        try:
            p_id = await state.add_prospect(prospect)
            logger.info(f"  + Prospect: {lead['contact_name']} ({lead['title']}) at {company_name}")
        except Exception as e:
            logger.warning(f"  * Prospect {lead['email']} note: {e}")

    logger.info("Successfully seeded all Growech customer profiles!")


if __name__ == "__main__":
    asyncio.run(main())
