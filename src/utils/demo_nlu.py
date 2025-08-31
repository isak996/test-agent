import re
SAFE={'拒答','不支持','安全拦截','闲聊'}

def _norm(q): return re.sub(r'\s+','',q.lower())

def predict_intent(query:str, context:str=None):
    q=_norm(query)
    if any(k in q for k in ['笑话','聊天','你喜欢','今天天气']):
        return {'intent':'闲聊_通用','confidence':0.6,'top_k':['闲聊_通用','播放_音乐']}
    if any(k in q for k in ['电台','广播']):
        return {'intent':'播放_电台','confidence':0.8,'top_k':['播放_电台','播放_音乐']}
    if any(k in q for k in ['音乐','来点','切到']):
        return {'intent':'播放_音乐','confidence':0.85,'top_k':['播放_音乐','播放_电台']}
    if any(k in q for k in ['带我去','导航到','去']):
        if any(p in q for p in ['医院','加油站','停车场','咖啡店','便利店']):
            return {'intent':'导航_POI','confidence':0.8,'top_k':['导航_POI','闲聊_通用']}
    if '空调' in q:
        if '调到' in q or '设定' in q: return {'intent':'车控_空调_设定温度','confidence':0.8,'top_k':['车控_空调_设定温度','车控_空调_开启']}
        return {'intent':'车控_空调_开启','confidence':0.75,'top_k':['车控_空调_开启','车控_空调_设定温度']}
    if any(k in q for k in ['焦虑症','失眠','咳嗽','减肥能不能']):
        return {'intent':'拒答','confidence':0.99,'top_k':['拒答','闲聊']}
    if any(k in q for k in ['蓝牙','胎压','语音唤醒','导航']):
        return {'intent':'用车_FAQ','confidence':0.7,'top_k':['用车_FAQ','闲聊_通用']}
    if any(k in q for k in ['生成','图片','壁纸','小游戏','挑战']):
        return {'intent':'不支持','confidence':0.9,'top_k':['不支持','拒答']}
    return {'intent':'不支持','confidence':0.5,'top_k':['不支持','闲聊']}
