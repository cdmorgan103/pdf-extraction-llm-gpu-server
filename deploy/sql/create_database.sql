/*
create_database.sql - prepare the network SQL Server for the pipeline.

Run once by a SQL Server administrator (SSMS, or:
    sqlcmd -S SQLSERVER01 -E -i deploy\sql\create_database.sql -v ServiceAccount="DOMAIN\svc_ocr"
).

Creates the database and gives the pipeline's service account only what it needs:
read and write data, and create its own tables (the pipeline creates and extends its
tables itself on first run, see src/db.py). It gets no server-wide rights.

Readers of the dashboard data (analysts, Power BI) can be added to db_datareader only.
*/

IF DB_ID(N'ocr_pipeline') IS NULL
    CREATE DATABASE ocr_pipeline;
GO

-- Windows (domain) account the scheduled tasks run as.
IF SUSER_ID(N'$(ServiceAccount)') IS NULL
    CREATE LOGIN [$(ServiceAccount)] FROM WINDOWS;
GO

USE ocr_pipeline;
GO

IF USER_ID(N'$(ServiceAccount)') IS NULL
    CREATE USER [$(ServiceAccount)] FOR LOGIN [$(ServiceAccount)];
ALTER ROLE db_datareader ADD MEMBER [$(ServiceAccount)];
ALTER ROLE db_datawriter ADD MEMBER [$(ServiceAccount)];
ALTER ROLE db_ddladmin   ADD MEMBER [$(ServiceAccount)];   -- create/alter its own tables
GO

-- Recommended: back up the database daily (full) as part of the server's backup plan,
-- e.g. a SQL Agent job:  BACKUP DATABASE ocr_pipeline TO DISK = N'<backup path>\ocr_pipeline.bak' WITH INIT;
