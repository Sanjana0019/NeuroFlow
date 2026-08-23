import asyncpg


async def check_schema(pool: asyncpg.Pool) -> bool:
    """Check whether the NeuroFlow database schema is present and apply minor migrations idempotently."""
    async with pool.acquire() as connection:
        has_schema = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'documents'
            )
            """
        )
        if not has_schema:
            return False

        # Idempotent migration for Task 8 columns and tables
        await connection.execute(
            """
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS description TEXT;
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active';
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

            CREATE TABLE IF NOT EXISTS pipeline_versions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                pipeline_id UUID NOT NULL REFERENCES pipelines(id) ON DELETE CASCADE,
                version INT NOT NULL,
                config JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (pipeline_id, version)
            );

            ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS pipeline_version INT NOT NULL DEFAULT 1;
            ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS fine_tuned_model TEXT;
            """
        )
        return True