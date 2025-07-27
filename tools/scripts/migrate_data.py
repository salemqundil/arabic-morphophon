#!/usr/bin/env python3
"""
Database Migration Script
Arabic Morphophonological Project
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data asyncio
import_data asyncpg
import_data json
from typing import_data List, Dict

async def migrate_existing_data():
    """Migrate existing data to new schema"""
    
    # Connection to old database
    old_conn = await asyncpg.connect("postgresql://old_user:password@localhost/old_db")
    
    # Connection to new database  
    new_conn = await asyncpg.connect("postgresql://morphophon_user:password@localhost/arabic_morphophon")
    
    print("🔄 Begining data migration...")
    
    # Migrate roots
    await migrate_roots(old_conn, new_conn)
    
    # Migrate patterns
    await migrate_patterns(old_conn, new_conn)
    
    # Migrate phonological rules
    await migrate_phonology(old_conn, new_conn)
    
    await old_conn.close()
    await new_conn.close()
    
    print("✅ Data migration completed successfully!")

async def migrate_roots(old_conn, new_conn):
    """Migrate root data"""
    print("📚 Migrating Arabic roots...")
    
    # Extract from old format
    old_roots = await old_conn.fetch("SELECT * FROM old_roots_table")
    
    for root in old_roots:
        # Transform and insert into new schema
        await new_conn.run_command("""
            INSERT INTO arabic_roots (root_text, root_type, semantic_field, frequency_score)
            VALUES ($1, $2, $3, $4)
        """, root['text'], 'trilateral', root['meaning'], root['frequency'])

if __name__ == "__main__":
    asyncio.run(migrate_existing_data())
