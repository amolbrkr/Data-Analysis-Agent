import streamlit as st
import pandas as pd
from openai import OpenAI
import duckdb
import plotly.express as px
import sqlalchemy
import json


class DataChatApp:
    def __init__(self):
        # Initialize session state
        if 'database_connection' not in st.session_state:
            st.session_state.database_connection = None
        if 'tables_info' not in st.session_state:
            st.session_state.tables_info = {}
        if 'database_type' not in st.session_state:
            st.session_state.database_type = None
        
        self.client = OpenAI(api_key=st.secrets["openai_key"])
        print("🚀 Initialized DataChatApp")
        
        # Initialize messages with enhanced structure if not exists
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Message structure will be:
        # {
        #     "role": "user" | "assistant",
        #     "content": str,
        #     "display_type": "text" | "sql" | "dataframe" | "error" | "plot",
        #     "data": Optional[Any]  # Contains dataframe, plot object, etc.
        # }
        
        # Updated tools schema with single function
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_and_execute_sql",
                    "description": "Generate and execute a SQL query based on a user's question. This function handles both query generation and execution.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_input": {
                                "type": "string",
                                "description": "The user's question or request about the data."
                            }
                        },
                        "required": ["user_input"],
                        "additionalProperties": False
                    }
                }
            }
        ]
            
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
            CREATE TABLE {quote}{table_name}{quote} ({',\n'.join(columns)});
            
            -- Sample data for {table_name}:
            {sample_data}"""
            schema_statements.append(create_table)
        
        final_schema = "\n".join(schema_statements)
        print(f"📋 Generated schema:\n{final_schema}")
        return final_schema

    def generate_and_execute_sql(self, user_input):
        """Generates SQL query from natural language input and executes it"""
        try:
            print(f"🤔 Processing user question: {user_input}")
            
            # Check if database is ready
            if not st.session_state.tables_info:
                return {
                    'type': 'error',
                    'content': "No tables available. Please upload data first."
                }

            # Generate SQL query
            schema_prompt = self._create_schema_prompt()
            quote_style = 'double quotes (")' if st.session_state.database_type == 'duckdb' else 'backticks (`)'
            
            prompt = f"""Given the following SQL tables, your job is to write queries given a user's request.

            {schema_prompt}

            Important notes:
            1. Use {quote_style} around table and column names to handle special characters
            2. Always fully qualify column names with table names
            3. The query must use only the tables and columns shown above
            4. Use proper table aliases if needed

            User Question: {user_input}

            Write a SQL query to answer this question. The query should work with {'PostgreSQL' if st.session_state.database_type == 'postgres' else 'DuckDB'} syntax.

            SQL Query:"""

            print(f"🔄 Sending prompt to OpenAI:\n{prompt}")
            
            # Get SQL query from OpenAI
            messages = [
                {
                    "role": "system", 
                    "content": "You are a SQL expert. Generate only the SQL query without any explanations."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )
            
            # Validate OpenAI response
            if not response.choices or not response.choices[0].message.content:
                return {
                    'type': 'error',
                    'content': "Failed to generate SQL query"
                }
            
            query = response.choices[0].message.content.strip()
            print(f"✨ Generated SQL query:\n{query}")
            
            # Execute the query
            try:
                print(f"▶️ Executing query:\n{query}")
                result = st.session_state.database_connection.execute(query).fetch_df()
                print(f"✅ Query executed successfully. Result shape: {result.shape}")
                
                return {
                    'type': 'sql',
                    'query': query,
                    'results': result
                }
                
            except Exception as exec_error:
                error_msg = str(exec_error)
                print(f"❌ Query execution failed: {error_msg}")
                
                # Enhanced error messages
                if "column not found" in error_msg.lower() or "not found in from clause" in error_msg.lower():
                    available_columns = []
                    for table_name, info in st.session_state.tables_info.items():
                        available_columns.extend([f"`{table_name}`.`{col}`" for col in info['columns']])
                    error_msg = f"Column not found. Available columns are: {', '.join(available_columns)}"
                
                return {
                    'type': 'error',
                    'query': query,
                    'content': f"Error executing query: {error_msg}"
                }
                
        except Exception as e:
            print(f"❌ Error in generate_and_execute_sql: {str(e)}")
            return {
                'type': 'error',
                'content': str(e)
            }

    def generate_visualization(self, data, viz_type='auto', column=None):
        """Generates visualization based on data and type"""
        print(f"📊 Generating {viz_type} visualization for {column}")
        try:
            if viz_type == 'histogram':
                return px.histogram(data, x=column, title=f'Distribution of {column}')
            elif viz_type == 'auto':
                # Determine appropriate visualization based on data
                if len(data.columns) == 1:
                    return px.histogram(data, x=data.columns[0], title=f'Distribution of {data.columns[0]}')
                elif len(data.columns) == 2:
                    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                    if len(numeric_cols) == 2:
                        return px.scatter(data, x=data.columns[0], y=data.columns[1])
                    else:
                        return px.bar(data, x=data.columns[0], y=data.columns[1])
            
            return None
        except Exception as e:
            print(f"❌ Error generating visualization: {str(e)}")
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
                table_schema = f"""CREATE TABLE {table_name} ({',\n    '.join(columns)});"""
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

    def process_user_input(self, user_input):
        """Process user input using OpenAI function calling"""
        try:
            print(f"🤔 Processing user input: {user_input}")
            
            if not st.session_state.tables_info:
                return {
                    'type': 'text',
                    'content': "No tables available. Please upload data first."
                }

            context = self.get_tables_info_text()
            messages = [
                {
                    "role": "system",
                    "content": """You are a data analysis assistant. Your job is to help users analyze their data by:
                1. Understanding their questions about the data
                2. Using SQL queries to answer their questions
                3. Providing information about the data structure when asked
                
                When a user's question requires data analysis, calculations, or filtering, 
                use the generate_and_execute_sql function to help answer their question."""
                },
                {
                    "role": "user",
                    "content": f"Current database state:\n{context}\n\nUser question: {user_input}"
                }
            ]

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0
            )

            message = response.choices[0].message
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                
                # Only one function to handle now
                return self.generate_and_execute_sql(function_args['user_input'])
            else:
                return {
                    'type': 'text',
                    'content': message.content if message.content else "No response generated"
                }
                    
        except Exception as e:
            print(f"❌ Error in process_user_input: {str(e)}")
            return {'type': 'error', 'content': str(e)}


def main():
    st.title("Dana: The Interactive Data Assistant")
    
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
            # Display content based on type
            if message["display_type"] == "text":
                st.markdown(message["content"])
                
            elif message["display_type"] == "sql":
                # Display SQL query
                st.markdown(message["content"])
                # Display results dataframe
                if message.get("data") is not None:
                    st.dataframe(message["data"])
                    
            elif message["display_type"] == "error":
                st.error(message["content"])
                
            elif message["display_type"] == "plot":
                st.markdown(message["content"])
                if message.get("data") is not None:
                    st.plotly_chart(message["data"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "display_type": "text",
            "data": None
        })
        
        # Process the input
        response = app.process_user_input(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            if response['type'] == 'sql':
                content = f"""I'll help you with that query!
```sql
{response['query']}
```
Here are the results:"""
                st.markdown(content)
                st.dataframe(response['results'])
                
                # Save message with both text and dataframe
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "display_type": "sql",
                    "data": response['results']
                })
                
            elif response['type'] == 'text':
                st.markdown(response['content'])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['content'],
                    "display_type": "text",
                    "data": None
                })
                
            else:  # error
                st.error(response['content'])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['content'],
                    "display_type": "error",
                    "data": None
                })


if __name__ == "__main__":
    main()
