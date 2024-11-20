import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from typing import Dict, Tuple, Optional, Union
import sqlalchemy
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.utilities import SQLDatabase


class DataChatApp:
    def __init__(self):
        # Initialize session state
        if 'database_connection' not in st.session_state:
            st.session_state.database_connection = None
        if 'tables_info' not in st.session_state:
            st.session_state.tables_info = {}
        if 'database_type' not in st.session_state:
            st.session_state.database_type = None
        
        self.llm = ChatOpenAI(temperature=0)
        print("🚀 Initialized DataChatApp")
        
        # Initialize messages in session state if not present
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
    def connect_to_database(self, connection_type, **kwargs):
        """Establishes database connection"""
        try:
            print(f"📡 Attempting to connect to {connection_type} database")
            if connection_type == "duckdb":
                st.session_state.database_connection = duckdb.connect(database=':memory:', read_only=False)
                st.session_state.database_type = "duckdb"
            elif connection_type == "postgres":
                connection_string = f"postgresql://{kwargs['user']}:{kwargs['password']}@{kwargs['host']}:{kwargs['port']}/{kwargs['database']}"
                print(f"🔌 Connecting to PostgreSQL at {kwargs['host']}:{kwargs['port']}/{kwargs['database']}")
                engine = sqlalchemy.create_engine(connection_string)
                st.session_state.database_connection = engine
                st.session_state.database_type = "postgres"
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            return False

    def load_file_to_database(self, uploaded_file):
        """Loads uploaded file into database"""
        try:
            print(f"📂 Processing uploaded file: {uploaded_file.name}")
            file_name = uploaded_file.name.split('.')[0]
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                print(f"❌ Unsupported file format: {file_extension}")
                return False, "Unsupported file format"
            
            print(f"📊 Loaded dataframe with shape: {df.shape}")
            st.session_state.database_connection.register(file_name, df)
            
            st.session_state.tables_info[file_name] = {
                'columns': list(df.columns),
                'shape': df.shape,
                'preview': df.head(),
                'dtypes': df.dtypes.to_dict()
            }
            
            self._update_schema_string()
            print(f"✅ Successfully loaded {file_name} into database")
            return True, file_name

        except Exception as e:
            print(f"❌ File loading failed: {str(e)}")
            return False, str(e)

    def _get_sql_type(self, dtype):
        """Convert pandas dtype to SQL type"""
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'FLOAT'
        elif 'datetime' in dtype_str:
            return 'DATETIME'
        else:
            return 'VARCHAR'

    def _create_schema_prompt(self):
        """Creates a schema prompt with CREATE TABLE statements and sample data"""
        print("🔍 Generating schema prompt")
        schema_statements = []
        
        # Use appropriate quotes based on database type
        quote = '"' if st.session_state.database_type == 'duckdb' else '`'
        
        for table_name, info in st.session_state.tables_info.items():
            # Add sample data preview
            preview_df = info['preview']
            sample_data = preview_df.to_string(index=False, max_rows=3)
            
            columns = []
            for col, dtype in info['dtypes'].items():
                sql_type = self._get_sql_type(dtype)
                columns.append(f'    {quote}{col}{quote} {sql_type}')
            
            create_table = f"""
    CREATE TABLE {quote}{table_name}{quote} (
{',\n'.join(columns)}
    );
    
    -- Sample data for {table_name}:
    {sample_data}"""
            schema_statements.append(create_table)
        
        final_schema = "\n".join(schema_statements)
        print(f"📋 Generated schema:\n{final_schema}")
        return final_schema

    def generate_sql(self, user_input):
        """Generates SQL query from natural language input"""
        try:
            print(f"🤔 Processing user question: {user_input}")
            
            if not st.session_state.tables_info:
                print("❌ No tables available in database")
                return "No tables available. Please upload data first."

            schema_prompt = self._create_schema_prompt()
            
            # Determine quote style based on database type
            quote_style = 'double quotes (")' if st.session_state.database_type == 'duckdb' else 'backticks (`)'
            
            prompt = f"""Given the following SQL tables, your job is to write queries given a user's request.

{schema_prompt}

Important notes:
1. Use {quote_style} around table and column names to handle special characters
2. Always fully qualify column names with table names
3. The query must use only the tables and columns shown above
4. Use proper table aliases if needed

User Question: {user_input}

Write a SQL query to answer this question. Use only the tables and columns provided above.
The query should work with {'PostgreSQL' if st.session_state.database_type == 'postgres' else 'DuckDB'} syntax.

SQL Query:"""

            print(f"🔄 Sending prompt to OpenAI:\n{prompt}")
            
            messages = [
                {
                    "role": "system", 
                    "content": f"""You are a SQL expert. Generate only the SQL query without any explanations.
                    Always use {quote_style} around table and column names.
                    Always fully qualify column names with table names or aliases."""
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.invoke(messages)
            query = response.content.strip()
            print(f"✨ Generated SQL query:\n{query}")
            
            if not query.lower().startswith('select'):
                print("❌ Invalid query generated - doesn't start with SELECT")
                return "Error: Generated query doesn't appear to be valid SQL"
            
            return query
            
        except Exception as e:
            print(f"❌ Error in SQL generation: {str(e)}")
            return f"Error generating SQL: {str(e)}"

    def execute_query(self, query):
        """Executes SQL query and returns results"""
        try:
            print(f"▶️ Attempting to execute query:\n{query}")
            
            # Print current tables and their columns for debugging
            print("📊 Available tables and columns:")
            for table_name, info in st.session_state.tables_info.items():
                print(f"Table: {table_name}")
                print(f"Columns: {info['columns']}")
            
            result = st.session_state.database_connection.execute(query).fetch_df()
            print(f"✅ Query executed successfully. Result shape: {result.shape}")
            return result
        except Exception as e:
            print(f"❌ Query execution failed: {str(e)}")
            error_msg = str(e)
            
            # Enhanced error messages
            if "column not found" in error_msg.lower() or "not found in from clause" in error_msg.lower():
                print("🔍 Column name issue detected. Checking available columns...")
                available_columns = []
                for table_name, info in st.session_state.tables_info.items():
                    available_columns.extend([f"`{table_name}`.`{col}`" for col in info['columns']])
                return f"Error: Column not found. Available columns are: {', '.join(available_columns)}"
            
            return f"Error executing query: {error_msg}"

    def generate_visualization(self, data):
        """Generates appropriate visualization based on data"""
        try:
            print("📊 Attempting to generate visualization")
            if data.empty:
                print("❌ No data available for visualization")
                return None
            
            # Determine appropriate visualization based on data
            if len(data.columns) == 2:
                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) == 1:
                    print("📊 Generating bar chart")
                    return px.bar(data, x=data.columns[0], y=data.columns[1])
                elif len(numeric_cols) == 2:
                    print("📊 Generating scatter plot")
                    return px.scatter(data, x=data.columns[0], y=data.columns[1])
            
            print("ℹ️ No suitable visualization type found")
            return None
        except Exception as e:
            print(f"❌ Visualization generation failed: {str(e)}")
            return None

    def _update_schema_string(self):
        """
        Updates the schema string used for SQL generation
        """
        print("🔄 Updating schema string")
        try:
            schema = []
            for table_name, info in st.session_state.tables_info.items():
                print(f"📝 Processing schema for table: {table_name}")
                
                # Get column definitions with types
                columns = []
                for col, dtype in info['dtypes'].items():
                    sql_type = self._get_sql_type(dtype)
                    columns.append(f"{col} {sql_type}")
                
                # Create table schema string
                table_schema = f"""CREATE TABLE {table_name} (
    {',\n    '.join(columns)}
);"""
                schema.append(table_schema)
            
            # Store the complete schema string in session state
            st.session_state.table_schemas = "\n\n".join(schema)
            print(f"✅ Updated schema string:\n{st.session_state.table_schemas}")
            
        except Exception as e:
            print(f"❌ Error updating schema string: {str(e)}")
            st.session_state.table_schemas = ""

    def get_tables_info_text(self):
        """Generate natural language description of loaded tables"""
        if not st.session_state.tables_info:
            return "There are no tables currently loaded in the database."
        
        info_text = []
        info_text.append(f"There are {len(st.session_state.tables_info)} tables currently loaded:")
        
        for table_name, info in st.session_state.tables_info.items():
            rows, cols = info['shape']
            info_text.append(f"\n- Table '{table_name}' with {rows:,} rows and {cols} columns:")
            info_text.append(f"  Columns: {', '.join(info['columns'])}")
            info_text.append(f"  Sample data:\n{info['preview'].to_string()}\n")
        
        return "\n".join(info_text)

    def generate_natural_response(self, user_input):
        """Generate natural language response for non-SQL questions"""
        try:
            print(f"🗣️ Generating natural language response for: {user_input}")
            
            # Create context about current database state
            context = self.get_tables_info_text()
            
            prompt = f"""Current database state:
{context}

User question: {user_input}

Please provide a natural language response about the database state. Be specific about the actual tables and data present."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful database assistant. Provide clear, concise responses about the database state and contents."
                },
                {"role": "user", "content": prompt}
            ]
            
            print(f"🔄 Sending prompt to OpenAI:\n{prompt}")
            response = self.llm.invoke(messages)
            print(f"✨ Generated response:\n{response.content}")
            
            return response.content
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def process_user_input(self, user_input):
        """Process user input and determine whether to generate SQL or natural response"""
        # Keywords that suggest a SQL query is needed
        sql_keywords = ['calculate', 'average', 'sum', 'count', 'show me', 'what is the', 'find', 'list', 'select']
        
        # Check if input seems to be asking for SQL query
        needs_sql = any(keyword in user_input.lower() for keyword in sql_keywords)
        
        if needs_sql:
            query = self.generate_sql(user_input)
            if query.startswith('Error'):
                return {'type': 'error', 'content': query}
            
            results = self.execute_query(query)
            if isinstance(results, str):  # Error occurred
                return {'type': 'error', 'content': results}
            
            return {
                'type': 'sql',
                'query': query,
                'results': results
            }
        else:
            response = self.generate_natural_response(user_input)
            return {
                'type': 'text',
                'content': response
            }


def main():
    st.title("Interactive Data Chat Application")
    
    app = DataChatApp()
    
    # Database Connection Section
    st.sidebar.header("Database Connection")
    connection_type = st.sidebar.radio(
        "Select Connection Type", ["File Upload", "PostgreSQL"]
    )

    if connection_type == "File Upload":
        uploaded_files = st.sidebar.file_uploader(
            "Upload CSV or Excel files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            app.connect_to_database("duckdb")
            for file in uploaded_files:
                success, message = app.load_file_to_database(file)
                if success:
                    st.sidebar.success(f"Loaded: {message}")
                else:
                    st.sidebar.error(message)

    else:
        # PostgreSQL connection form
        with st.sidebar.form("postgres_connection"):
            host = st.text_input("Host")
            port = st.text_input("Port", "5432")
            database = st.text_input("Database")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Connect"):
                if app.connect_to_database(
                    "postgres",
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                ):
                    st.sidebar.success("Connected to PostgreSQL!")

    # Display table information in sidebar
    if st.session_state.tables_info:
        st.sidebar.header("Available Tables")
        for table_name, info in st.session_state.tables_info.items():
            with st.sidebar.expander(f"📊 {table_name}"):
                st.write("**Preview:**")
                st.dataframe(info["preview"])
                st.write(
                    f"**Shape:** {info['shape'][0]} rows, {info['shape'][1]} columns"
                )
                st.write("**Columns:**", ", ".join(info["columns"]))

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process the input
        response = app.process_user_input(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            if response['type'] == 'sql':
                st.markdown("I'll help you with that query!")
                st.code(response['query'], language='sql')
                st.markdown("Here are the results:")
                st.dataframe(response['results'])
                
                # Generate visualization if applicable
                fig = app.generate_visualization(response['results'])
                if fig:
                    st.plotly_chart(fig)
                    
            elif response['type'] == 'text':
                st.markdown(response['content'])
            else:  # error
                st.error(response['content'])
        
        # Save assistant response to chat history
        content = response.get('content', response.get('query', 'Error processing request'))
        st.session_state.messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
