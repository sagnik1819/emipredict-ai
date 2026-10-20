"""
Database and CRUD Operations Service for EMIPredict AI.
Provides persistent storage, querying, auditing, and lifecycle management
for loan applicants using an ACID-compliant SQLite backend.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from .config import DB_PATH

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the applicants table schema and indices."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS applicants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applicant_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                marital_status TEXT NOT NULL,
                education TEXT NOT NULL,
                monthly_salary REAL NOT NULL,
                employment_type TEXT NOT NULL,
                years_of_employment INTEGER NOT NULL,
                company_type TEXT NOT NULL,
                house_type TEXT NOT NULL,
                monthly_rent REAL NOT NULL,
                family_size INTEGER NOT NULL,
                dependents INTEGER NOT NULL,
                school_fees REAL NOT NULL,
                college_fees REAL NOT NULL,
                travel_expenses REAL NOT NULL,
                groceries_utilities REAL NOT NULL,
                other_monthly_expenses REAL NOT NULL,
                existing_loans INTEGER NOT NULL,
                current_emi_amount REAL NOT NULL,
                credit_score INTEGER NOT NULL,
                bank_balance REAL NOT NULL,
                emergency_fund REAL NOT NULL,
                emi_scenario TEXT NOT NULL,
                requested_amount REAL NOT NULL,
                requested_tenure INTEGER NOT NULL,
                predicted_eligibility TEXT,
                predicted_max_emi REAL,
                application_status TEXT DEFAULT 'Pending',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scenario ON applicants(emi_scenario)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON applicants(application_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eligibility ON applicants(predicted_eligibility)")
        conn.commit()

class ApplicantService:
    """Service class executing business CRUD operations."""

    @staticmethod
    def create(data: Dict[str, Any]) -> int:
        init_db()
        now = datetime.now().isoformat()
        fields = [
            "applicant_name", "age", "gender", "marital_status", "education",
            "monthly_salary", "employment_type", "years_of_employment", "company_type",
            "house_type", "monthly_rent", "family_size", "dependents",
            "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
            "other_monthly_expenses", "existing_loans", "current_emi_amount",
            "credit_score", "bank_balance", "emergency_fund", "emi_scenario",
            "requested_amount", "requested_tenure", "predicted_eligibility",
            "predicted_max_emi", "application_status", "notes", "created_at", "updated_at"
        ]
        
        row_values = [data.get(f, None) for f in fields[:-2]] + [now, now]
        placeholders = ", ".join(["?"] * len(fields))
        columns = ", ".join(fields)

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"INSERT INTO applicants ({columns}) VALUES ({placeholders})", row_values)
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_id(applicant_id: int) -> Optional[Dict[str, Any]]:
        init_db()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM applicants WHERE id = ?", (applicant_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_all(
        search: Optional[str] = None,
        scenario: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        init_db()
        query = "SELECT * FROM applicants WHERE 1=1"
        params = []

        if search:
            query += " AND (applicant_name LIKE ? OR id LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        if scenario and scenario != "All":
            query += " AND emi_scenario = ?"
            params.append(scenario)
        if status and status != "All":
            query += " AND application_status = ?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    @staticmethod
    def update(applicant_id: int, updates: Dict[str, Any]) -> bool:
        init_db()
        updates["updated_at"] = datetime.now().isoformat()
        set_clauses = [f"{k} = ?" for k in updates.keys()]
        values = list(updates.values()) + [applicant_id]

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE applicants SET {', '.join(set_clauses)} WHERE id = ?", values)
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def delete(applicant_id: int) -> bool:
        init_db()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM applicants WHERE id = ?", (applicant_id,))
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def get_portfolio_summary() -> Dict[str, Any]:
        init_db()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(requested_amount), AVG(monthly_salary), AVG(credit_score) FROM applicants")
            row = cursor.fetchone()
            total_count = row[0] or 0
            avg_amount = row[1] or 0.0
            avg_salary = row[2] or 0.0
            avg_cs = row[3] or 0.0

            cursor.execute("SELECT predicted_eligibility, COUNT(*) FROM applicants GROUP BY predicted_eligibility")
            elig_counts = dict(cursor.fetchall())

            return {
                "total_records": total_count,
                "avg_loan_requested": round(avg_amount, 2),
                "avg_monthly_salary": round(avg_salary, 2),
                "avg_credit_score": round(avg_cs, 1),
                "eligibility_distribution": elig_counts
            }

    @staticmethod
    def seed_initial_records(df_sample: pd.DataFrame):
        """Seeds initial realistic applicant records if table is empty."""
        init_db()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM applicants")
            if cursor.fetchone()[0] == 0:
                print("Seeding initial records into applicant records database...")
                sample = df_sample.head(150).copy()
                names = [
                    "Aarav Sharma", "Priya Patel", "Rohan Mehta", "Ananya Iyer", "Vikram Singh",
                    "Sneha Reddy", "Arjun Nair", "Kavita Rao", "Deepak Gupta", "Meera Joshi",
                    "Aditya Verma", "Pooja Deshmukh", "Rajesh Kumar", "Neha Bansal", "Siddharth Jain"
                ]
                import random
                for idx, row in sample.iterrows():
                    full_name = f"{random.choice(names)} #{idx+1}"
                    ApplicantService.create({
                        "applicant_name": full_name,
                        "age": int(row["age"]),
                        "gender": str(row["gender"]),
                        "marital_status": str(row["marital_status"]),
                        "education": str(row["education"]),
                        "monthly_salary": float(row["monthly_salary"]),
                        "employment_type": str(row["employment_type"]),
                        "years_of_employment": int(row["years_of_employment"]),
                        "company_type": str(row["company_type"]),
                        "house_type": str(row["house_type"]),
                        "monthly_rent": float(row["monthly_rent"]),
                        "family_size": int(row["family_size"]),
                        "dependents": int(row["dependents"]),
                        "school_fees": float(row["school_fees"]),
                        "college_fees": float(row["college_fees"]),
                        "travel_expenses": float(row["travel_expenses"]),
                        "groceries_utilities": float(row["groceries_utilities"]),
                        "other_monthly_expenses": float(row["other_monthly_expenses"]),
                        "existing_loans": int(row["existing_loans"]),
                        "current_emi_amount": float(row["current_emi_amount"]),
                        "credit_score": int(row["credit_score"]),
                        "bank_balance": float(row["bank_balance"]),
                        "emergency_fund": float(row["emergency_fund"]),
                        "emi_scenario": str(row["emi_scenario"]),
                        "requested_amount": float(row["requested_amount"]),
                        "requested_tenure": int(row["requested_tenure"]),
                        "predicted_eligibility": str(row.get("emi_eligibility", "Eligible")),
                        "predicted_max_emi": float(row.get("max_monthly_emi", 15000.0)),
                        "application_status": random.choice(["Approved", "Under Review", "Pending", "Flagged"]),
                        "notes": "Automated ingestion during initialization"
                    })
                print("Seeding complete.")
