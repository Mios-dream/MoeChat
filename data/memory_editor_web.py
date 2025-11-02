import os
import sys
from datetime import datetime, time
from ruamel.yaml import YAML
import traceback
import uuid as uuid_generator # 用于生成核心记忆ID
import time as time_module    # 用于生成长期记忆ID
import re

from flask import Flask, render_template, request, redirect, url_for, flash

#路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_MEM_FILE = os.path.join(BASE_DIR, "core_mem.yml")
LONG_MEM_DIR = os.path.join(BASE_DIR, "memorys")

#MemoryEditor
class MemoryEditor:
    """
    一个离线的记忆后端。
    负责所有文件的 I/O (加载, 保存, 删除)。
    """
    def __init__(self):
        self.db = []
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.width = 4096

        if not os.path.exists(CORE_MEM_FILE) or not os.path.exists(LONG_MEM_DIR):
            print("[错误] 记忆文件或目录未找到。")
            raise FileNotFoundError("记忆文件或目录未找到。请检查路径配置。")

        self.load_all_mems()

    def load_all_mems(self):
        """加载所有记忆"""
        self.db.clear()
        error_count = 0
        try:
            error_count += self._load_core_mem()
            error_count += self._load_long_mem()
            self.db.sort(key=lambda x: (self.get_datetime_from_entry(x) is None, self.get_datetime_from_entry(x)))
            print(f"--- 记忆加载完成，共 {len(self.db)} 条。加载过程中遇到 {error_count} 个警告。---")
        except Exception as e:
            print(f"[严重错误] 加载记忆时发生错误: {e}")
            traceback.print_exc()
            self.db.clear()

    def _load_core_mem(self) -> int:
        """加载核心记忆, 返回警告数量"""
        warnings = 0
        try:
            # 文件是否存在
            if not os.path.exists(CORE_MEM_FILE):
                 print(f"[警告] 核心记忆文件未找到: {CORE_MEM_FILE}")
                 return warnings # 或者返回 0

            with open(CORE_MEM_FILE, 'r', encoding='utf-8') as f:
                data = self.yaml.load(f)
            if data:
                for uuid, entry in data.items():
                    if isinstance(entry, dict) and 'text' in entry and 'time' in entry:
                        self.db.append({
                            'type': 'core', 'id': uuid, 'file': os.path.basename(CORE_MEM_FILE), 'data': entry
                        })
                    elif isinstance(uuid, str) and uuid.startswith('#'): pass
                    else:
                         print(f"[警告] 跳过核心记忆中格式不正确的条目: ID={uuid}")
                         warnings += 1
        except Exception as e:
            print(f"[错误] 加载核心记忆 {CORE_MEM_FILE} 失败: {e}")
            raise
        return warnings

    def _load_long_mem(self) -> int:
        """加载长期记忆, 返回警告数量"""
        warnings = 0
        if not os.path.exists(LONG_MEM_DIR):
             print(f"[警告] 长期记忆目录未找到: {LONG_MEM_DIR}")
             return warnings

        for root, _, files in os.walk(LONG_MEM_DIR):
            for file in files:
                if file.endswith(('.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = self.yaml.load(f)
                        if data:
                            for timestamp, entry in data.items():
                                if isinstance(timestamp, int) and isinstance(entry, dict) and 'msg' in entry and 'text_tag' in entry:
                                    self.db.append({
                                        'type': 'long', 'id': timestamp, 'file': file, 'data': entry
                                    })
                                elif isinstance(timestamp, str) and timestamp.startswith('#'): pass
                                else:
                                    print(f"[警告] 跳过长期记忆文件 {file} 中格式不正确的条目: ID={timestamp}")
                                    warnings += 1
                    except Exception as e:
                        print(f"[错误] 加载长期记忆 {file} 失败: {e}")
        return warnings


    def get_datetime_from_entry(self, entry):
        try:
            if entry.get('type') == 'core':
                return datetime.strptime(entry['data']['time'], '%Y-m-d %H:%M:%S')
            elif entry.get('type') == 'long':
                return datetime.fromtimestamp(int(entry['id']))
        except (ValueError, TypeError, KeyError, OSError):
            return None

    def get_unique_dates(self) -> list[str]:
        """获取唯一日期列表"""
        dates = set()
        for entry in self.db:
            dt = self.get_datetime_from_entry(entry)
            if dt:
                dates.add(dt.strftime('%Y-%m-%d'))
        sorted_dates = sorted(list(dates))
        return ["-- 所有日期 --"] + sorted_dates

    def find_entry_by_id_and_type(self, entry_id, entry_type):
        """根据 ID 和类型查找内存中的条目"""
        target_id = entry_id
        if entry_type == 'long':
            try:
                target_id = int(entry_id)
            except (ValueError, TypeError):
                return None

        for entry in self.db:
            if entry.get('type') == entry_type and entry.get('id') == target_id:
                return entry
        return None


    def search(self, keyword, start_date_str, end_date_str):
                """统一搜索功能 (搜索标签和所有对话行)"""
                results = self.db
                kw_lower = keyword.lower() if keyword else None # 预处理关键词

                # --- 关键词过滤 ---
                if kw_lower:
                    temp_results = []
                    for entry in results:
                        match = False
                        if entry.get('type') == 'core':
                            text = entry['data'].get('text', '').lower()
                            if kw_lower in text:
                                match = True
                        elif entry.get('type') == 'long':
                            msg = entry['data'].get('msg', '')
                            text_tag = entry['data'].get('text_tag', '').lower()
                            
                            # 1. 搜索标签
                            if text_tag and kw_lower in text_tag:
                                match = True
                            else:
                                # 2. 搜索所有对话行 (跳过时间戳行)
                                lines = msg.split('\n')
                                if len(lines) > 1:
                                    # 搜索第2行及之后的所有内容
                                    content_to_search = "\n".join(lines[1:]).lower()
                                    if kw_lower in content_to_search:
                                        match = True
                        if match:
                            temp_results.append(entry)
                    results = temp_results

                # --- 日期过滤
                start_dt, end_dt = None, None
                if start_date_str and start_date_str != "-- 所有日期 --":
                    try: start_dt = datetime.combine(datetime.strptime(start_date_str, '%Y-%m-%d'), time.min)
                    except ValueError: pass
                if end_date_str and end_date_str != "-- 所有日期 --":
                    try: end_dt = datetime.combine(datetime.strptime(end_date_str, '%Y-%m-%d'), time.max)
                    except ValueError: pass
                if start_dt or end_dt:
                    results = [
                        e for e in results
                        if (dt := self.get_datetime_from_entry(e))
                        and (not start_dt or dt >= start_dt)
                        and (not end_dt or dt <= end_dt)
                    ]
                return results

    def _save_yaml_data(self, file_path, data):
        """封装 YAML 保存逻辑"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(data, f)
        except Exception as e:
            raise IOError(f"写入文件 {file_path} 失败: {e}")

    def _delete_pkl(self, entry):
        """封装 PKL 删除逻辑"""
        if entry.get('type') == 'long':
            filename = entry.get('file')
            if filename:
                pkl_file = filename.replace('.yaml', '.pkl').replace('.yml', '.pkl')
                pkl_path = os.path.join(LONG_MEM_DIR, pkl_file)
                if os.path.exists(pkl_path):
                    try: os.remove(pkl_path); print(f"[调试] 删除了 {pkl_path}")
                    except Exception as e: print(f"[错误] 删除 {pkl_path} 失败: {e}")

    def save_entry(self, entry):
        """保存修改或新增的条目"""
        file_path = self._get_file_path(entry)
        if not file_path: raise ValueError("无法确定文件路径")

        data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f: data = self.yaml.load(f)
                if data is None: data = {}
            except Exception as e:
                print(f"[警告] 读取文件 {file_path} 失败，将尝试覆盖: {e}")
                data = {}

        entry_id = entry.get('id')
        entry_data = entry.get('data')
        if entry_id is None or entry_data is None: raise ValueError("条目缺少 ID 或数据")

        data[entry_id] = entry_data
        self._save_yaml_data(file_path, data)
        self._delete_pkl(entry)

    def delete_entry(self, entry):
        """删除一个条目"""
        file_path = self._get_file_path(entry)
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"无法找到要删除的文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f: data = self.yaml.load(f)
        if data is None: data = {}

        entry_id = entry.get('id')
        if entry_id is None: raise ValueError("要删除的条目缺少 ID")

        if entry_id in data:
            del data[entry_id]
            self._save_yaml_data(file_path, data)
            self._delete_pkl(entry)
        else:
            print(f"[警告] 尝试删除不存在的条目 ID: {entry_id} in file {file_path}")
    
    def _get_file_path(self, entry):
        """获取条目的完整文件路径"""
        entry_type = entry.get('type')
        if entry_type == 'core':
            return CORE_MEM_FILE
        elif entry_type == 'long':
            filename = entry.get('file')
            if filename: return os.path.join(LONG_MEM_DIR, filename)
        return None


# Flask Web


app = Flask(__name__)
app.secret_key = os.urandom(24)


from datetime import datetime as dt # 避免与 datetime 模块本身冲突

@app.template_filter('strftime')
def _jinja2_filter_datetime(date, fmt=None):
    """Jinja2 filter for formatting dates/timestamps."""
    if date is None:
        return "N/A" # 处理 None 值

    # 尝试将 int/float 时间戳转换为 datetime
    if isinstance(date, (int, float)):
        try:
            date = dt.fromtimestamp(date)
        except (OSError, ValueError): # 处理无效时间戳
             return "Invalid Timestamp"

    if isinstance(date, dt):
        # 如果 fmt 为 None 或空字符串，使用默认格式
        format_string = fmt if fmt else '%Y-%m-%d %H:%M:%S'
        try:
            # 移除可能的时区信息，strftime 不处理带时区的对象
            native_date = date.replace(tzinfo=None)
            return native_date.strftime(format_string)
        except ValueError:
             return "Invalid Date Format" # 处理格式化错误

    # 如果不是有效的日期/时间对象，尝试将其作为字符串返回
    return str(date)



# --- 初始化 MemoryEditor
try:
    editor = MemoryEditor()
except FileNotFoundError as e:
    print(f"[严重] Flask 应用启动失败: {e}")
    sys.exit(1)
# ---------------------------------------------

@app.context_processor
def inject_editor():
     """将 editor 对象注入到所有模板的上下文中，以便模板可以直接调用其方法"""
     return dict(editor=editor)

@app.route('/')
def index():
    """主页路由，显示搜索栏和结果"""
    results = []
    unique_dates = editor.get_unique_dates()
    return render_template('index.html', results=results, unique_dates=unique_dates)

@app.route('/search')
def search():
    """处理搜索请求"""
    keyword = request.args.get('keyword', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    try:
        results = editor.search(keyword, start_date, end_date)
        unique_dates = editor.get_unique_dates()
        
        for entry in results:
            if entry.get('type') == 'long':
                entry['generated_preview'] = _generate_long_mem_preview(entry)
                
        flash(f"搜索到 {len(results)} 条结果。", "info")
    except Exception as e:
        flash(f"搜索时发生错误: {e}", "error")
        results = []
        unique_dates = ["-- 所有日期 --"]

    return render_template('index.html',
                           results=results,
                           unique_dates=unique_dates,
                           keyword=keyword,
                           start_date=start_date,
                           end_date=end_date)

@app.route('/edit')
def edit_form():
    """显示编辑表单"""
    entry_id = request.args.get('id')
    entry_type = request.args.get('type')

    if not entry_id or not entry_type:
        flash("缺少条目 ID 或类型", "error")
        return redirect(url_for('index'))

    entry = editor.find_entry_by_id_and_type(entry_id, entry_type)

    if not entry:
        flash(f"未找到要编辑的条目 (ID: {entry_id}, 类型: {entry_type})", "error")
        return redirect(url_for('index'))

    return render_template('edit.html', entry=entry)

@app.route('/save', methods=['POST'])
def save_changes():
    """处理编辑表单的提交"""
    try:
        entry_id_str = request.form['id']
        entry_type = request.form['type']
        original_file = request.form.get('file', '')

        updated_data = {}
        original_entry = editor.find_entry_by_id_and_type(entry_id_str, entry_type)

        if not original_entry:
             raise KeyError(f"无法找到原始条目 (ID: {entry_id_str}, 类型: {entry_type}) 以进行保存")


        if entry_type == 'core':
            entry_id = entry_id_str
            updated_data['text'] = request.form['text']
            updated_data['time'] = original_entry['data']['time']
        elif entry_type == 'long':
            entry_id = int(entry_id_str)
            updated_data['text_tag'] = request.form['text_tag']
            updated_data['msg'] = request.form['msg']
        else:
            raise ValueError("未知的条目类型")

        updated_entry = {
            'id': entry_id,
            'type': entry_type,
            'file': original_file,
            'data': updated_data
        }

        editor.save_entry(updated_entry)
        editor.load_all_mems()
        flash("记忆保存成功！", "success")

    except (ValueError, IOError, KeyError, Exception) as e:
        flash(f"保存记忆时出错: {e}", "error")
        traceback.print_exc()

    return redirect(url_for('index'))


@app.route('/delete', methods=['POST'])
def delete_item():
    """处理删除请求"""
    try:
        entry_id_str = request.form['id']
        entry_type = request.form['type']


        entry_to_delete = editor.find_entry_by_id_and_type(entry_id_str, entry_type)
        if not entry_to_delete:
             raise KeyError(f"无法找到要删除的条目 (ID: {entry_id_str}, 类型: {entry_type})")


        editor.delete_entry(entry_to_delete)
        editor.load_all_mems() 
        flash("记忆删除成功！", "success")

    except (FileNotFoundError, ValueError, IOError, KeyError, Exception) as e:
        flash(f"删除记忆时出错: {e}", "error")
        traceback.print_exc()

    return redirect(url_for('index'))

def _generate_long_mem_preview(entry: dict) -> str:
    """
    【新版】辅助函数：自动解析并生成 long 记忆的预览字符串。
    返回包含 <br> 标签的 HTML 字符串，用于前端显示。
    """
    msg = entry.get('data', {}).get('msg', '')
    lines = msg.split('\n')
    
    user_line = None
    ai_line = None

    # 自动获取第2行 (用户) 和 第3行 (AI) 的完整内容
    if len(lines) > 1:
        user_line = lines[1].strip() 

    if len(lines) > 2:
        ai_line = lines[2].strip() 

    # 构建预览字符串
    parts_list = []
    if user_line: 
        parts_list.append(user_line)
    if ai_line: 
        parts_list.append(ai_line)
    
    # 空行
    preview = "<br><br>".join(parts_list)

    # 备选方案
    if not preview and len(lines) > 1:
        preview = lines[1].strip() # 使用第2行原文
    

    # preview = preview[:80] + ('...' if len(preview) > 80 else '')
    
    return preview


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)