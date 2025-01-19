


CUSTOM_LLM_AND_EMBED=False #True #False #控制query阶段是否使用自定义的llm和embed模型。
#使用方法：index阶段改成true，query阶段改成false
#现存一些问题：
# 1、如果True的话，哪怕在inde阶段也会加载模型。→ 因为graphrag.index阶段使用的vector_stores（包含自定义更改）是和query阶段公用的
# 2、改成False同时要把final_entities.parquet中的description_embeddings改成openai.text_embedding_ada_002的1536维度


import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

#不管上面的CUDA_VISIBLE_DEVICES写什么，这里只能从0开始写。实际部署在哪个id的卡上，还是要看CUDA_VISIBLE_DEVICES的值
QWEN2_5_7B_INSTRUCT_DEVICE_MAP={
    "lm_head": 1,
    "model.embed_tokens": 0,
    "model.layers.0":0,
    "model.layers.1":0,
    "model.layers.2":0,
    "model.layers.3":0,
    "model.layers.4":0,
    "model.layers.5":0,
    "model.layers.6":0,
    "model.layers.7":0,
    "model.layers.8":0,
    "model.layers.9":0,
    "model.layers.10":0,
    "model.layers.11":0,
    "model.layers.12":0,
    "model.layers.13":0,
    "model.layers.14":1,
    "model.layers.15":1,
    "model.layers.16":1,
    "model.layers.17":1,
    "model.layers.18":1,
    "model.layers.19":1,
    "model.layers.20":1,
    "model.layers.21":1,
    "model.layers.22":1,
    "model.layers.23":1,
    "model.layers.24":1,
    "model.layers.25":1,
    "model.layers.26":1,
    "model.layers.27":1,   
    "model.norm":1,
}





def load_custom_embed():
    model = SentenceTransformer('/mnt/afs/chenxiaoxuan/hf_home/KaLM-embedding-multilingual-mini-instruct-v1.5')   # Do NOT set trust_remote_code
    model.max_seq_length = 512
    return model



def loda_custom_llm():
    model_name='/mnt/afs/chenxiaoxuan/hf_home/Qwen2.5-7B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" #QWEN2_5_7B_INSTRUCT_DEVICE_MAP
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer



def llm_generate(model,tokenizer,query_prompt,system_prompt):
        
    #prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def cxx_get_embedding(model,sentences):
    embeddings = model.encode(
        sentences, 
        normalize_embeddings=True,
        batch_size=256, 
        show_progress_bar=True
        )
    return embeddings

# 只加载一次模型
if CUSTOM_LLM_AND_EMBED==True:
    _embedding_model = load_custom_embed()
    _custom_llm,_custom_tokenizer = loda_custom_llm()
    print("Model loaded.")
else:
    _embedding_model = None
    _custom_llm = None
    _custom_tokenizer = None
    
'''
import time
print(time.time())
r=llm_generate(_custom_llm,_custom_tokenizer,"""\n---Role---\n\nYou are a helpful assistant responding to questions about data in the tables provided.\n\n\n---Goal---\n\nGenerate a response of the target length and format that responds to the user\'s question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.\n\nIf you don\'t know the answer, just say so. Do not make anything up.\n\nPoints supported by data should list their data references as follows:\n\n"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."\n\nDo not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.\n\nFor example:\n\n"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."\n\nwhere 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.\n\nDo not include information where the supporting evidence for it is not provided.\n\n\n---Target response length and format---\n\nMultiple Paragraphs\n\n\n---Data tables---\n\n-----Reports-----\nid|title|content\n35|Electricity Bill Payment and Customer Service Hotlines|"# Electricity Bill Payment and Customer Service Hotlines\n\nThe community revolves around the process of paying electricity bills and the associated customer service hotlines. Key entities include the electricity bill itself, the events of making payments and deductions, and the customer service hotlines for WeChat (95017) and Alipay (95188). These entities are interconnected through various relationships that facilitate the payment process and provide customer support.\n\n## Electricity Bill as the central entity\n\nThe electricity bill (电费) is the central entity in this community, representing the cost of electricity consumed by customers. It is the focal point around which other entities and events revolve. The importance of the electricity bill is underscored by its high degree of connections with other entities and events, such as payment and deduction processes, and customer service hotlines [Data: Entities (665); Relationships (333, 619, 115, 997, 1020, +more)].\n\n## Role of customer service hotlines\n\nCustomer service hotlines play a crucial role in this community by providing support for issues related to electricity bill payments. The 95017 hotline is associated with WeChat services, while the 95188 hotline is linked to Alipay services. These hotlines assist customers with payment issues, deductions, and general inquiries, highlighting their importance in ensuring smooth transactions and resolving customer concerns [Data: Entities (680, 681); Relationships (338, 339, 1021, 1022, 1024, 1025, 1026, 1027)].\n\n## Payment and deduction events\n\nThe events of making payments (交费) and deductions (扣款) are integral to the electricity bill payment process. These events are facilitated by various platforms and hotlines, ensuring that customers can pay their bills efficiently. The 95598 hotline, for example, assists with both payment and deduction issues, demonstrating the interconnectedness of these events with customer service support [Data: Entities (688, 689); Relationships (620, 621, 1020)].\n\n## State Grid Online platform\n\nThe State Grid Online platform (网上国网) is another significant entity in this community, providing a digital means for customers to pay their electricity bills. This platform\'s role in the payment process highlights the increasing reliance on digital solutions for managing essential services, further emphasizing the importance of seamless and reliable payment systems [Data: Relationships (115)].\n\n## E-Power Treasure app\n\nThe E-Power Treasure app (电E宝) is used for paying electricity bills, representing another digital solution within this community. The app\'s integration into the payment process underscores the trend towards mobile and app-based solutions for managing utility payments, which can enhance convenience for customers but also necessitates robust customer support systems [Data: Relationships (997)]."\n\n\n-----Entities-----\nid|entity|description|number of relationships\n688|交费|The event of making a payment for electricity bills|3\n659|电量电费页|The page within the WeChat public account for paying electricity bills|1\n151|电量电价||1\n652|掌上电力|掌上电力是一种移动应用，提供电费交费服务|1\n689|扣款|The event of deducting money from a customer\'s account for electricity payments|3\n670|公摊电费|公摊电费 is a type of property public service fee collected by property service enterprises for public electricity usage within a property management area|0\n682|营业厅柜台|The service counter at the business hall where customers can pay their electricity bills|1\n665|电费|电费 (Electricity Bill) refers to the cost of electricity consumed by an enterprise, unit, or individual over a certain period. It encompasses the monthly electricity charges that customers need to pay. Essentially, it is the event of paying electricity bills.|7\n684|生活缴费|The WeChat Life Payment service used for paying utility bills|1\n646|供电分离移交|供电分离移交是指将国有企业职工家属区的供电设施移交给供电公司|3\n380|单相本地费控电能表||1\n675|电度电费|电度电费 is the electricity fee calculated based on the customer\'s actual electricity consumption and the approved electricity price, including catalog electricity fee and surcharges|0\n138|电度单价查询方法||2\n434|互供电价||1\n816|电费发票|电费发票 is an event where an electricity bill is issued to a customer. Additionally, 电费发票 is one of the valid documents that a customer can use to prove their identity.|2\n690|供电方||8\n201|并网电能计量表|并网电能计量表用于发电量统计和电价补贴|1\n643|办公|小区内的办公用电设施需要进行移交和改造|1\n409|智能电表|智能电表是一种先进的电能表，能够通过远程通信技术实时传输电能数据。供电公司将统一安装智能电表，并与用户签订智能缴费协议。|3\n841|欠费停电|Power outage due to unpaid bills|1\n\n\n-----Relationships-----\nid|source|target|description|weight|rank|links\n1028|营业厅柜台|供电方|The electricity provider manages payments through the service counter at the business hall|6.0|9|1\n427|供电公司|智能电表|供电公司将统一安装智能电表，并与用户签订智能缴费协议|8.0|174|4\n433|供电公司|供电分离移交|供电公司负责供电分离移交项目中的供电设施接管和改造|8.0|174|4\n514|供电公司|电费发票|电费发票是客户可以用来证明身份的有效证件之一|6.0|173|4\n525|供电公司|欠费停电|供电公司 handles power outages due to unpaid bills|8.0|172|4\n618|95598|供电方|The electricity provider\'s customer service can be reached via the 95598 hotline|7.0|36|4\n619|95598|电费|The 95598 hotline assists with issues related to electricity payments|7.0|35|4\n620|95598|交费|The 95598 hotline assists with issues related to making payments for electricity bills|7.0|31|4\n621|95598|扣款|The 95598 hotline assists with issues related to deductions for electricity payments|7.0|31|4\n340|客户|供电方|The electricity provider supplies electricity to the customer and manages their payments|9.0|91|3\n333|客户|电费|Customers use WeChat to pay their monthly electricity bills|1.0|90|3\n357|客户|电费发票|Customers need to provide an electricity bill for power-related services|14.0|85|3\n114|网上国网|供电方|The electricity provider manages payments through the State Grid Online platform|6.0|25|3\n115|网上国网|电费|State Grid Online platform is used for paying electricity bills|7.0|24|3\n101|网上国网|电度单价查询方法|The online platform provides a method for querying electricity unit prices|7.0|19|3\n1021|电费|95017|The 95017 hotline assists with issues related to WeChat payments|7.0|12|3\n1022|电费|95188|The 95188 hotline assists with issues related to Alipay payments|7.0|12|3\n1024|95017|交费|The 95017 hotline assists with issues related to making payments for WeChat services|7.0|8|3\n1025|95017|扣款|The 95017 hotline assists with issues related to deductions for WeChat payments|7.0|8|3\n1026|95188|交费|The 95188 hotline assists with issues related to making payments for Alipay services|7.0|8|3\n1027|95188|扣款|The 95188 hotline assists with issues related to deductions for Alipay payments|1.0|8|3\n280|供电企业|智能电表|供电企业将统一安装智能电表，并与用户签订智能缴费协议|8.0|43|2\n286|供电企业|掌上电力|掌上电力提供供电企业的电费交费服务|6.0|41|2\n1015|微信|供电方|The electricity provider manages payments through WeChat|6.0|20|2\n996|电E宝|供电方|The electricity provider manages payments through the E-Power Treasure app|6.0|16|2\n992|移交企业|供电分离移交|移交企业负责供电分离移交项目中的协调和配合工作|7.0|15|2\n997|电E宝|电费|E-Power Treasure app is used for paying electricity bills|7.0|15|2\n765|电能表|智能电表|智能电表是一种先进的电能表，能够通过远程通信技术实时传输电能数据|8.0|14|2\n991|移交企业|办公|移交企业负责小区内办公用电设施的移交和改造|1.0|13|2\n1016|微信|生活缴费|WeChat Life Payment service is used for paying utility bills|7.0|13|2\n759|电能表|单相本地费控电能表|单相本地费控电能表是一种电能表，具有费控功能|7.0|12|2\n194|分布式电源|并网电能计量表|并网电能计量表用于分布式电源的发电量统计和电价补贴|8.0|26|1\n1029|第三方|供电方|The electricity provider allows payments through third-party channels|6.0|15|1\n977|国有企业|供电分离移交|国有企业负责供电分离移交项目中的协调和配合工作|7.0|12|1\n1000|支付宝|供电方|The electricity provider manages payments through Alipay|6.0|12|1\n1020|电费|代扣|Automatic deductions are used to pay the electricity bill|1.0|10|1\n1007|网上国网95598微信公众号|电量电费页|The WeChat public account includes a page for paying electricity bills|7.0|10|1\n152|单一制电价|电量电价|Single-part electricity pricing is based on actual electricity consumption|8.0|7|1\n99|95598热线|电度单价查询方法|The 95598 hotline provides a method for querying electricity unit prices|7.0|4|1\n823|互供电力|互供电价|The exchange of electricity between different power grids is priced as inter-grid electricity|8.0|2|1\n\n\n\n\n-----Sources-----\nid|text\n59|业20%、居民生活用电80%，当月用电100度，电费为：100×20%×0.7370（一般工商业电价）+100×80%×0.4983（居民电价）=50.61元。\n交纳电费渠道介绍\n \n电费交费笔数及金额限制\n通过营业厅柜台交费：无笔数及金额限制。\n通过除营业厅柜台外各代收渠道（网上国网、微信、支付宝、银行、自助缴费终端、第三方）交费：\n1.供电方限制：\n笔数限制：每户每月交费（含预存）次数不超过60次，每日不超过5次。\n金额限制：每笔限额不超过5万元。\n2.渠道方限制：详见附件“代收渠道交费限制（陕西）”。\n \n客户交费成功但未到账的处理方式\n原东区与原南区无差异。\n1.微信交费未到账：首先确认客户是通过【生活缴费】还是微信公众号交费，公众号名称详见微信公众号介绍；\n（1）微信【生活缴费】主动交费：建议客户先联系微信在线客服或拨打微信服务热线95017确认是否为微信端故障，如非微信端故障，可拨打95598供电服务热线，客服专员根据相关业务规范进行处理；\n（2）微信【生活缴费】开通的自动扣款：此类客户是指已在网上国网APP、电e宝APP、微信生活交费进行签约成为自动扣款客户。一般当日19时前到账，最迟在扣款后2-3日到账，建议客户耐心等待。若扣款后3日仍未到账且未收到退款信息，可根据相应业务规则处理。\n（3）微信公众号：请客户第二天查询是否有退款信息。若没有退款信息，可先联系微信客服核实是否存在系统故障，若无故障问题，再致电95598，由客服专员通过系统查询，确认款项确实未到账后，记录客户户号、联系方式、交费或预存方式及时间，根据相关业务规范进行处理；\n2.支付宝交费未到账：\n支付宝【生活缴费】：请客户联系支付宝客服电话95188，由支付宝方处理；\n3.网上国网交费未到账：详见网上国网交费未到账处理方式。\n4.电e宝交费未到账：详见电e宝交费到账时间。\n注：客户在线上渠道（不包含网上国网APP）交费扣款成功，但一直显示交费中，在交费记录中查询不到该笔交费记录。出现该问题，第二天系统会自动补账或退款，补账后能即时查到该笔记录，退款会在7个工作日内原路返回；客户通过网上国网APP交费后出现此种问题的处理方式详见：网上国网交费未到账处理方式\n \n\n\n\n\n\n\n\n\n55|电子渠道\n网上国网95598微信公众号交纳电费方法\n进入网上国网95598微信公众号，点击底部菜单“去交费”进入电量电费页，通过微信小程序完成户号绑定进行交费。\n      \n网上国网95598微信公众号介绍\n1.\t在微信-通讯录-公众号-右上角“+”输入“网上国网95598”进行搜索，在搜索结果里点击“网上国网”后关注该公众号。\n2.关注后可交纳电费，查询电网检修信息以及咨询类信息。\n3.网上国网95598微信公众号无法查询电量电费、余额等信息，若想查询可前往网上国网APP进行查询。\n \n网上国网95598微信公众号绑定账号方式\n1.\t进入网上国网95598微信公众号首页，点击“我的服务”-“账号绑定”中完成账号绑定，绑定后无法解绑。\n2.账号绑定后可自动同步网上国网APP信息。\n3.网上国网95598微信公众号无法绑定户号，可进入微信小程序后在交费中添加户号，添加的户号不能同步至网上国网APP。\n4.网上国网95598微信公众号无法解绑户号，若想解绑已在网上国网95598微信公众号中显示的户号信息，可在网上国网APP中解绑，解绑后网上国网95598微信公众号会自动同步。\n  \n网上国网95598微信公众号推送异常\n1.\t网上国网95598微信公众号推送账单显示年份为2024年\n（1）异常原因：系统推送数据错误\n（2）引导方法：建议用户实际账单月份以营业厅清单为准，也可前往网上国网APP查询\n2.用户收到非本人户号的账单推送，需取消推送信息：\n记录用户手机号、推送的户号等派发工单至网上国网运营中心。\n网上国网95598微信公众号在线客服使用方法\n1.点击公众号下方【我的服务】，点击【在线客服】\n2.填写用户信息，选择服务省份\n3.成功接入在线客服，跳转至聊天页面。（如成功获取客户相关信息则点击【在线客服】直接跳转至聊天页面）\n4.可以在接入会话后查看近半年的聊天记录。\n网上国网95598微信公众号交费异常处理方法\n点击“首页-去交费，提示”对不起，当前页面无法访问“：\n异常原因：由于系统优化调整，影响此页面暂时无法访问。\n引导方法：建议用户直接点击公众号底部菜单”去交费“，或者下载网上国网APP进行交费。\n网上国网95598微信公众号查询电网检修信息方法\n进入网上国网95598微信公众号-我的服务-电网检修信息，即可进行查看。\n微信生活缴费问题处理\n1.\t如交费时遇到系统故障（如提示通信失败）等问题导致无法交费，建议客户联系微信在线客服或拨打微信服务热线95017确认是否为微信端故障，或选择其他方式交费。\n2.如客户咨询交费未到账问题，详见：客户交费成功但未到账的处理方式\n微信生活缴费签约代扣规则\n原东区与原南区无差异：\n1.一个户号仅可被一个微信号签约代扣，若该户号已被其他微信号签约过代扣，则无法再次签约；\n2.不同渠道的代扣影响微信签约代扣，例如客户已开通过银行、支付宝等渠道的代扣，不可以继续开通微信签约代扣；\n3.处于代扣在途的户号（通过其它渠道办理代扣业务），此期间不支持开通微信签约代扣。\n4.微信代扣签约时可设置预警金额，客户在签约界面可自由选择预警值（10元、30元、50元、100元、200元\n\n\n\n---Goal---\n\nGenerate a response of the target length and format that responds to the user\'s question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.\n\nIf you don\'t know the answer, just say so. Do not make anything up.\n\nPoints supported by data should list their data references as follows:\n\n"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."\n\nDo not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.\n\nFor example:\n\n"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."\n\nwhere 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.\n\nDo not include information where the supporting evidence for it is not provided.\n\n\n---Target response length and format---\n\nMultiple Paragraphs\n\nAdd sections and commentary to the response as appropriate for the length and format. Style the response in markdown.\n""",'电费怎么交')
print(time.time())
'''
#def get_custom_embed_model():
    #if _embedding_model==None:
    #    _embedding_model = load_custom_embed()
#    return _embedding_model

#def get_custom_llm_model():
    #if _custom_llm==None or _custom_tokenizer==None:
    #    assert _custom_llm==None and _custom_tokenizer==None
    #    _custom_llm,_custom_tokenizer = loda_custom_llm()
#    return _custom_llm,_custom_tokenizer

