import asyncio
import sqlite3
from harvey.integrations.email_finder import get_mx_host, verify_gemini

async def main():
    con = sqlite3.connect('C:/harvey/data/harvey.db')
    cur = con.cursor()
    rows = cur.execute("SELECT id, company, email, email_status FROM prospects WHERE email != '' AND email_status IN ('guess', 'risky')").fetchall()
    print(f"Checking {len(rows)} existing prospects with Gemini + MX...")
    
    for pid, company, email, status in rows:
        domain = email.partition('@')[2]
        mx = await get_mx_host(domain)
        if not mx:
            cur.execute("UPDATE prospects SET email_status = 'invalid' WHERE id = ?", (pid,))
            print(f"{email} -> INVALID (No MX)")
        else:
            res = await verify_gemini(email, mx_host=mx, company=company)
            if res and res.get('status') == 'valid':
                cur.execute("UPDATE prospects SET email_status = 'verified' WHERE id = ?", (pid,))
                print(f"{email} ({company}) -> VERIFIED by Gemini (MX: {mx})")
            elif res and res.get('status') == 'invalid':
                cur.execute("UPDATE prospects SET email_status = 'invalid' WHERE id = ?", (pid,))
                print(f"{email} -> INVALID by Gemini")
                
    con.commit()
    con.close()
    print("Done update.")

if __name__ == '__main__':
    asyncio.run(main())
