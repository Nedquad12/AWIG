import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext, ttk
import requests
import os
import time
import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

class UnifiedStockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Data Downloader & Calculator")
        self.root.geometry("900x700")
        
        # Initialize variables
        self.max_urls = 100
        self.url_entries = []
        self.current_page = 0
        self.urls_per_page = 10
        self.all_urls = [""] * self.max_urls
        self.downloaded_files = []
        self.calculation_results = []
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_download_tab()
        self.create_calculate_tab()
        
    def create_download_tab(self):
        # Download tab
        self.download_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.download_frame, text="Download Data")
        
        # Top frame for navigation
        top_frame = tk.Frame(self.download_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # URL entries frame
        self.entries_frame = tk.Frame(self.download_frame)
        self.entries_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Bottom frame for buttons
        bottom_frame = tk.Frame(self.download_frame)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Navigation controls
        tk.Label(top_frame, text="Page:").pack(side=tk.LEFT)
        self.page_var = tk.StringVar()
        self.update_page_display()
        page_label = tk.Label(top_frame, textvariable=self.page_var)
        page_label.pack(side=tk.LEFT, padx=5)
        
        prev_btn = tk.Button(top_frame, text="◀ Previous", command=self.prev_page)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = tk.Button(top_frame, text="Next ▶", command=self.next_page)
        next_btn.pack(side=tk.LEFT, padx=5)
        
        # Create URL entries
        self.create_url_entries()
        
        # Action buttons - First row
        button_row1 = tk.Frame(bottom_frame)
        button_row1.pack(fill=tk.X, pady=2)
        
        download_btn = tk.Button(button_row1, text="Download Selected", command=self.download_selected)
        download_btn.pack(side=tk.LEFT, padx=5)
        
        download_all_btn = tk.Button(button_row1, text="Download All", command=self.download_data)
        download_all_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(button_row1, text="Clear All", command=self.clear_all)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Action buttons - Second row
        button_row2 = tk.Frame(bottom_frame)
        button_row2.pack(fill=tk.X, pady=2)
        
        save_btn = tk.Button(button_row2, text="Save URLs", command=self.save_urls)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        load_btn = tk.Button(button_row2, text="Load URLs", command=self.load_urls)
        load_btn.pack(side=tk.LEFT, padx=5)
        
    def create_calculate_tab(self):
        # Calculate tab
        self.calculate_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.calculate_frame, text="Calculate Results")
        
        # Ticker input frame
        ticker_frame = tk.Frame(self.calculate_frame)
        ticker_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(ticker_frame, text="Enter Tickers (separated by comma):").pack(anchor='w')
        self.tickers_var = tk.StringVar()
        self.ticker_entry = tk.Entry(ticker_frame, textvariable=self.tickers_var, width=50)
        self.ticker_entry.pack(fill=tk.X, pady=5)
        
        # Buttons frame
        buttons_frame = tk.Frame(self.calculate_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.calculate_btn = tk.Button(buttons_frame, text="Calculate from Downloaded Files", 
                                      command=self.calculate_from_downloaded, state=tk.DISABLED)
        self.calculate_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_files_btn = tk.Button(buttons_frame, text="Load External CSV Files", 
                                       command=self.load_external_files)
        self.load_files_btn.pack(side=tk.LEFT, padx=5)
        
        self.download_results_btn = tk.Button(buttons_frame, text="Download Results (XLSX)", 
                                             command=self.download_results, state=tk.DISABLED)
        self.download_results_btn.pack(side=tk.LEFT, padx=5)
        
        # Results treeview
        self.result_tree = ttk.Treeview(self.calculate_frame, 
                                       columns=('Ticker', 'Quantity Total', 'Market Value Total'), 
                                       show='headings', height=15)
        self.result_tree.heading('Ticker', text='Ticker')
        self.result_tree.heading('Quantity Total', text='Quantity Total')
        self.result_tree.heading('Market Value Total', text='Market Value Total')
        self.result_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.calculate_frame, textvariable=self.status_var, 
                                    relief=tk.SUNKEN, anchor='w')
        self.status_label.pack(fill=tk.X, padx=10, pady=5)
        
    def create_url_entries(self):
        # Clear existing entries if any
        for widget in self.entries_frame.winfo_children():
            widget.destroy()
        
        self.url_entries = []
        self.checkbox_vars = []
        
        # Create header
        header_frame = tk.Frame(self.entries_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        select_all_var = tk.BooleanVar()
        select_all_cb = tk.Checkbutton(header_frame, text="Select All", variable=select_all_var, 
                                       command=lambda: self.select_all(select_all_var.get()))
        select_all_cb.pack(side=tk.LEFT)
        
        tk.Label(header_frame, text="URLs", width=70).pack(side=tk.LEFT, padx=5)
        
        # Create entries for current page
        start_idx = self.current_page * self.urls_per_page
        for i in range(self.urls_per_page):
            idx = start_idx + i
            if idx < self.max_urls:
                entry_frame = tk.Frame(self.entries_frame)
                entry_frame.pack(fill=tk.X, pady=2)
                
                # Checkbox for selection
                check_var = tk.BooleanVar()
                check = tk.Checkbutton(entry_frame, variable=check_var)
                check.pack(side=tk.LEFT)
                self.checkbox_vars.append(check_var)
                
                # URL number label
                tk.Label(entry_frame, text=f"URL {idx+1}:", width=8, anchor="w").pack(side=tk.LEFT)
                
                # URL entry field
                entry = tk.Entry(entry_frame, width=70)
                entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                if self.all_urls[idx]:
                    entry.insert(0, self.all_urls[idx])
                entry.bind("<FocusOut>", lambda event, idx=idx, entry=entry: self.save_entry(idx, entry))
                self.url_entries.append(entry)
    
    def save_entry(self, idx, entry):
        self.all_urls[idx] = entry.get()
    
    def select_all(self, value):
        for var in self.checkbox_vars:
            var.set(value)
    
    def update_page_display(self):
        total_pages = (self.max_urls + self.urls_per_page - 1) // self.urls_per_page
        self.page_var.set(f"{self.current_page+1} / {total_pages}")
    
    def next_page(self):
        self.save_current_entries()
        max_pages = (self.max_urls + self.urls_per_page - 1) // self.urls_per_page
        if self.current_page < max_pages - 1:
            self.current_page += 1
            self.update_page_display()
            self.create_url_entries()
    
    def prev_page(self):
        self.save_current_entries()
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page_display()
            self.create_url_entries()
    
    def save_current_entries(self):
        start_idx = self.current_page * self.urls_per_page
        for i, entry in enumerate(self.url_entries):
            idx = start_idx + i
            if idx < self.max_urls:
                self.all_urls[idx] = entry.get()
    
    def sanitize_filename(self, filename):
        invalid_chars = r'[\\/*?:"<>|]'
        return ''.join(c for c in filename if c not in invalid_chars)
    
    def extract_filename_from_url(self, url):
        start = url.find('fileName=') + len('fileName=')
        if start != -1:
            end = url.find('&', start)
            if end == -1:
                end = len(url)
            filename = url[start:end].split('_')[0]
            return filename
        return None
    
    def download_selected(self):
        self.save_current_entries()
        selected_indices = []
        start_idx = self.current_page * self.urls_per_page
        for i, var in enumerate(self.checkbox_vars):
            if var.get():
                selected_indices.append(start_idx + i)
        
        if not selected_indices:
            messagebox.showinfo("Info", "No URLs selected for download")
            return
        
        self.download_urls([self.all_urls[idx] for idx in selected_indices])
    
    def download_data(self):
        self.save_current_entries()
        urls_to_download = [url for url in self.all_urls if url]
        if not urls_to_download:
            messagebox.showinfo("Info", "No URLs to download")
            return
        
        self.download_urls(urls_to_download)
    
    def download_urls(self, urls):
        if not os.path.exists('download_data'):
            os.makedirs('download_data')
        
        errors = []
        successful = 0
        self.downloaded_files = []
        
        for i, url in enumerate(urls, 1):
            if url:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    filename = self.extract_filename_from_url(url)
                    if not filename:
                        filename = f"data_{int(time.time())}_{i}"
                    
                    filename = self.sanitize_filename(filename)
                    
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/csv' in content_type:
                        filename += '.csv'
                    elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
                        filename += '.xlsx'
                    elif 'application/vnd.ms-excel' in content_type:
                        filename += '.xls'
                    elif 'application/pdf' in content_type:
                        filename += '.pdf'
                    elif 'application/json' in content_type:
                        filename += '.json'
                    else:
                        filename += '.csv'  # Default to CSV for calculation
                    
                    full_path = os.path.join('download_data', filename)
                    with open(full_path, 'wb') as file:
                        file.write(response.content)
                    
                    # Add to downloaded files list if it's a CSV
                    if filename.endswith('.csv'):
                        self.downloaded_files.append(full_path)
                    
                    successful += 1
                except requests.RequestException as e:
                    errors.append(f"Failed to download URL {i}: {e}")
                except OSError as e:
                    errors.append(f"Error saving URL {i}: {e}")
        
        # Show results
        if successful > 0:
            messagebox.showinfo("Success", f"{successful} file(s) successfully downloaded to download_data folder")
            # Enable calculate button if we have downloaded CSV files
            if self.downloaded_files:
                self.calculate_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Downloaded {len(self.downloaded_files)} CSV files ready for calculation")
                # Switch to calculate tab
                self.notebook.select(1)
        
        if errors:
            messagebox.showerror("Errors", "\n".join(errors))
    
    def clear_all(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all URLs?"):
            self.all_urls = [""] * self.max_urls
            self.create_url_entries()
            messagebox.showinfo("Info", "All URLs have been cleared")
    
    def save_urls(self):
        self.save_current_entries()
        urls_to_save = [url for url in self.all_urls if url]
        
        if not urls_to_save:
            messagebox.showinfo("Info", "No URLs to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    json.dump(urls_to_save, file, indent=2)
                messagebox.showinfo("Success", f"{len(urls_to_save)} URLs saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save URLs: {e}")
    
    def load_urls(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as file:
                loaded_urls = json.load(file)
            
            if not isinstance(loaded_urls, list):
                messagebox.showerror("Error", "Invalid file format. Expected a list of URLs.")
                return
            
            action = self.show_load_options()
            
            if action == "edit":
                self.load_urls_for_editing(loaded_urls)
            elif action == "download":
                self.download_urls(loaded_urls)
        
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON file")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load URLs: {e}")
    
    def show_load_options(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Options")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="What would you like to do with the loaded URLs?", 
                 wraplength=280).pack(pady=10)
        
        result = [None]
        
        tk.Button(dialog, text="Load for Editing", 
                  command=lambda: self.set_result(dialog, result, "edit")).pack(pady=5)
        
        tk.Button(dialog, text="Download Now", 
                  command=lambda: self.set_result(dialog, result, "download")).pack(pady=5)
        
        tk.Button(dialog, text="Cancel", 
                  command=lambda: self.set_result(dialog, result, "cancel")).pack(pady=5)
        
        self.root.wait_window(dialog)
        return result[0]
    
    def set_result(self, dialog, result_container, value):
        result_container[0] = value
        dialog.destroy()
    
    def load_urls_for_editing(self, loaded_urls):
        num_loaded = min(len(loaded_urls), self.max_urls)
        self.all_urls = [""] * self.max_urls
        
        for i in range(num_loaded):
            self.all_urls[i] = loaded_urls[i]
        
        self.current_page = 0
        self.update_page_display()
        self.create_url_entries()
        
        messagebox.showinfo("Success", f"{num_loaded} URLs loaded for editing")
    
    def load_external_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            self.downloaded_files = list(file_paths)
            self.calculate_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Loaded {len(self.downloaded_files)} external CSV files")
            messagebox.showinfo("Info", f"Loaded {len(self.downloaded_files)} CSV files.")
    
    def calculate_from_downloaded(self):
        tickers_input = self.tickers_var.get().strip()
        if not tickers_input:
            messagebox.showerror("Error", "Please enter ticker symbols.")
            return
        
        if not self.downloaded_files:
            messagebox.showerror("Error", "No CSV files available for calculation.")
            return
        
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        self.calculation_results = []
        
        for ticker in tickers:
            total_quantity = 0
            total_market_value = 0
            
            for file_path in self.downloaded_files:
                try:
                    # Try to read CSV with different approaches
                    try:
                        # First try with skiprows (for the original format)
                        df = pd.read_csv(file_path, skiprows=9, 
                                       names=['Ticker', 'Name', 'Sector', 'Asset Class', 
                                             'Market Value', 'Weight (%)', 'Notional Value', 
                                             'Quantity', 'Price', 'Location', 'Exchange', 
                                             'Currency', 'FX Rate', 'Market Currency', 'Accrual Date'])
                    except:
                        # If that fails, try reading normally
                        df = pd.read_csv(file_path)
                    
                    # Clean and standardize ticker column
                    if 'Ticker' in df.columns:
                        df['Ticker'] = df['Ticker'].astype(str).str.upper()
                        df_ticker = df[df['Ticker'] == ticker]
                        
                        if not df_ticker.empty:
                            # Calculate totals with error handling
                            if 'Quantity' in df.columns:
                                quantity_col = df_ticker['Quantity'].fillna(0)
                                for val in quantity_col:
                                    try:
                                        clean_val = str(val).replace(',', '').replace('$', '')
                                        total_quantity += float(clean_val)
                                    except:
                                        continue
                            
                            if 'Market Value' in df.columns:
                                market_value_col = df_ticker['Market Value'].fillna(0)
                                for val in market_value_col:
                                    try:
                                        clean_val = str(val).replace(',', '').replace('$', '')
                                        total_market_value += float(clean_val)
                                    except:
                                        continue
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
            
            # Add to results if we found any data
            if total_quantity > 0 or total_market_value > 0:
                self.calculation_results.append({
                    'Ticker': ticker,
                    'Quantity Total': total_quantity,
                    'Market Value Total': total_market_value
                })
        
        self.update_results_treeview()
        self.download_results_btn.config(state=tk.NORMAL if self.calculation_results else tk.DISABLED)
        
        if self.calculation_results:
            self.status_var.set(f"Calculation complete. Found data for {len(self.calculation_results)} tickers.")
        else:
            self.status_var.set("No data found for the specified tickers.")
            messagebox.showinfo("Info", "No data found for the specified tickers in the CSV files.")
    
    def update_results_treeview(self):
        # Clear existing items
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Add results
        for result in self.calculation_results:
            self.result_tree.insert('', 'end', values=(
                result['Ticker'],
                f"{result['Quantity Total']:,.2f}",
                f"${result['Market Value Total']:,.2f}"
            ))
    
    def download_results(self):
        if not self.calculation_results:
            messagebox.showwarning("Warning", "No results to download.")
            return
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Stock Calculation Results"
        
        df_results = pd.DataFrame(self.calculation_results)
        
        for r in dataframe_to_rows(df_results, index=False, header=True):
            ws.append(r)
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", 
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if file_path:
            wb.save(file_path)
            messagebox.showinfo("Success", f"Results saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedStockApp(root)
    root.mainloop()