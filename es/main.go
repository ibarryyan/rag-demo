package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

// 配置结构体
type Config struct {
	ElasticHost    string
	ElasticPort    int
	DeepSeekAPIKey string
	DeepSeekModel  string
	IndexName      string
}

// 文档结构体
type Document struct {
	ID      string                 `json:"id"`
	Title   string                 `json:"title"`
	Content string                 `json:"content"`
	Vector  []float32              `json:"vector,omitempty"`
	Meta    map[string]interface{} `json:"meta,omitempty"`
}

// 搜索结果
type SearchResult struct {
	Title   string  `json:"title"`
	Content string  `json:"content"`
	Score   float64 `json:"score"`
}

// RAG系统
type RAGSystem struct {
	elasticClient *elasticsearch.Client
	openAIClient  *openai.Client
	config        Config
}

func main() {
	fmt.Println("🚀 ElasticSearch 8.x RAG Demo启动...")
	fmt.Println("=====")

	// 加载配置
	config := loadConfig()

	// 创建RAG系统
	rag, err := NewRAGSystem(config)
	if err != nil {
		log.Fatalf("创建RAG系统失败: %v", err)
	}
	defer func() {
		_ = rag.elasticClient.Close(context.Background())
	}()

	// 初始化知识库
	fmt.Println("\n📚 正在初始化知识库...")
	err = rag.InitializeKnowledgeBase()
	if err != nil {
		log.Fatalf("初始化知识库失败: %v", err)
	}
	fmt.Println("✅ 知识库初始化完成")

	// 测试问题
	testQuestions := []string{
		"闫同学是谁？",
		"介绍一下扯编程的淡公众号",
	}

	// 运行对比测试
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("🧪 开始对比测试")
	fmt.Println(strings.Repeat("=", 50))

	for i, question := range testQuestions {
		fmt.Printf("\n📝 测试 %d/%d\n", i+1, len(testQuestions))
		fmt.Printf("❓ 问题: %s\n", question)

		// 获取直接答案
		fmt.Println("\n🔍 获取纯DeepSeek回答...")
		directAnswer, directTime, err := rag.GetDirectAnswer(question)
		if err != nil {
			fmt.Printf("❌ 获取直接答案失败: %v\n", err)
			continue
		}
		fmt.Printf("⏱️  响应时间: %.2f秒\n", directTime)
		fmt.Printf("💬 回答: %s\n", directAnswer)

		// 获取RAG答案
		fmt.Println("\n🔍 获取RAG增强回答...")
		ragAnswer, ragTime, sources, err := rag.GetRAGAnswer(question)
		if err != nil {
			fmt.Printf("❌ 获取RAG答案失败: %v\n", err)
			continue
		}
		fmt.Printf("⏱️  响应时间: %.2f秒\n", ragTime)
		fmt.Printf("💬 回答: %s\n", ragAnswer)

		// 显示检索到的文档
		if len(sources) > 0 {
			fmt.Println("\n📄 检索到的相关文档:")
			for j, source := range sources {
				fmt.Printf("  %d. [相似度: %.2f] %s\n", j+1, source.Score, source.Title)
				if j == 0 { // 只显示最相关文档的片段
					content := source.Content
					if len(content) > 100 {
						content = content[:100] + "..."
					}
					fmt.Printf("     内容: %s\n", content)
				}
			}
		}

		// 简单对比分析
		fmt.Println("\n📊 对比分析:")
		fmt.Printf("  - 时间开销: RAG比纯DeepSeek慢 %.2f秒\n", ragTime-directTime)
		fmt.Printf("  - 信息质量: RAG基于 %d 个相关文档生成\n", len(sources))

		if i < len(testQuestions)-1 {
			fmt.Println("\n" + strings.Repeat("-", 50))
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("🎉 测试完成!")
	fmt.Println("💡 总结: ElasticSearch RAG在需要混合搜索的场景表现更好")
	fmt.Println(strings.Repeat("=", 50))
}

// 加载配置
func loadConfig() Config {
	godotenv.Load()

	return Config{
		ElasticHost:    getEnv("ELASTIC_HOST", "localhost"),
		ElasticPort:    getEnvAsInt("ELASTIC_PORT", 9200),
		DeepSeekAPIKey: getEnv("DEEPSEEK_API_KEY", ""),
		DeepSeekModel:  getEnv("DEEPSEEK_MODEL", "deepseek-chat"),
		IndexName:      getEnv("INDEX_NAME", "rag_documents"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvAsInt(key string, defaultValue int) int {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	var result int
	fmt.Sscanf(value, "%d", &result)
	return result
}

// 创建RAG系统
func NewRAGSystem(config Config) (*RAGSystem, error) {
	if config.DeepSeekAPIKey == "" {
		return nil, fmt.Errorf("DEEPSEEK_API_KEY不能为空")
	}

	// 连接ElasticSearch 8.x
	elasticURL := fmt.Sprintf("http://%s:%d", config.ElasticHost, config.ElasticPort)
	cfg := elasticsearch.Config{
		Addresses: []string{elasticURL},
	}

	client, err := elasticsearch.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("连接ElasticSearch失败: %w", err)
	}

	// 测试连接
	res, err := client.Info()
	if err != nil {
		return nil, fmt.Errorf("测试ElasticSearch连接失败: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return nil, fmt.Errorf("ElasticSearch连接错误: %s", res.String())
	}

	// 创建OpenAI客户端
	conf := openai.DefaultConfig(config.DeepSeekAPIKey)
	conf.BaseURL = "https://api.deepseek.com"

	return &RAGSystem{
		elasticClient: client,
		openAIClient:  openai.NewClientWithConfig(conf),
		config:        config,
	}, nil
}

// 初始化知识库
func (r *RAGSystem) InitializeKnowledgeBase() error {
	indexName := r.config.IndexName

	// 检查索引是否存在
	res, err := r.elasticClient.Indices.Exists([]string{indexName})
	if err != nil {
		return fmt.Errorf("检查索引存在失败: %w", err)
	}
	defer res.Body.Close()

	// 如果索引存在，先删除（为了演示）
	if res.StatusCode == 200 {
		res, err := r.elasticClient.Indices.Delete([]string{indexName})
		if err != nil {
			return fmt.Errorf("删除索引失败: %w", err)
		}
		defer res.Body.Close()

		if res.IsError() {
			return fmt.Errorf("删除索引错误: %s", res.String())
		}
	}

	// 创建索引 mapping - ElasticSearch 8.x 格式
	mapping := map[string]interface{}{
		"settings": map[string]interface{}{
			"number_of_shards":   1,
			"number_of_replicas": 0,
			"analysis": map[string]interface{}{
				"analyzer": map[string]interface{}{
					"default": map[string]interface{}{
						"type": "standard",
					},
				},
			},
		},
		"mappings": map[string]interface{}{
			"properties": map[string]interface{}{
				"id": map[string]interface{}{
					"type": "keyword",
				},
				"title": map[string]interface{}{
					"type":     "text",
					"analyzer": "standard",
				},
				"content": map[string]interface{}{
					"type":     "text",
					"analyzer": "standard",
				},
				"vector": map[string]interface{}{
					"type":       "dense_vector",
					"dims":       4,
					"index":      true,
					"similarity": "cosine",
				},
				"meta": map[string]interface{}{
					"type":    "object",
					"dynamic": true,
				},
				"timestamp": map[string]interface{}{
					"type": "date",
				},
			},
		},
	}

	// 序列化mapping为JSON
	mappingJSON, err := json.Marshal(mapping)
	if err != nil {
		return fmt.Errorf("序列化mapping失败: %w", err)
	}

	// 创建索引
	res, err = r.elasticClient.Indices.Create(
		indexName,
		r.elasticClient.Indices.Create.WithBody(bytes.NewReader(mappingJSON)),
	)
	if err != nil {
		return fmt.Errorf("创建索引失败: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("创建索引错误: %s", res.String())
	}

	// 插入示例文档
	err = r.insertSampleDocuments()
	if err != nil {
		return fmt.Errorf("插入文档失败: %w", err)
	}

	// 等待索引刷新
	res, err = r.elasticClient.Indices.Refresh(
		r.elasticClient.Indices.Refresh.WithIndex(indexName),
	)
	if err != nil {
		return fmt.Errorf("刷新索引失败: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("刷新索引错误: %s", res.String())
	}

	fmt.Printf("✅ 索引 %s 创建成功\n", indexName)
	return nil
}

// 插入示例文档
func (r *RAGSystem) insertSampleDocuments() error {
	indexName := r.config.IndexName

	// 示例文档数据
	documents := []Document{
		{
			ID:      "doc_001",
			Title:   "闫同学人物介绍",
			Content: "闫同学，男，来自中国，26岁，天蝎座，是知名技术博主、摄影博主、技术爱好者，擅长写Go语言，喜欢打羽毛球。",
			Vector:  r.generateSimpleVector("闫同学人物介绍"),
			Meta: map[string]interface{}{
				"category": "人物介绍",
				"source":   "闫同学人物介绍",
				"date":     "2026-02-04",
			},
		},
		{
			ID:      "doc_002",
			Title:   "扯编程的淡公众号介绍",
			Content: "扯编程的淡，科技领域知名微信公众号，由闫同学运营，内容多为技术博客，日常生活感想，截止2026年1月，已有粉丝2000+。",
			Vector:  r.generateSimpleVector("扯编程的淡公众号介绍"),
			Meta: map[string]interface{}{
				"category": "公众号介绍",
				"source":   "扯编程的淡公众号介绍",
				"date":     "2026-02-04",
			},
		},
	}

	// 批量插入文档
	var bulkBuffer bytes.Buffer
	for _, doc := range documents {
		// 添加时间戳
		if doc.Meta == nil {
			doc.Meta = make(map[string]interface{})
		}
		doc.Meta["timestamp"] = time.Now()

		// 添加操作行
		meta := map[string]interface{}{
			"index": map[string]interface{}{
				"_index": indexName,
				"_id":    doc.ID,
			},
		}

		metaJSON, _ := json.Marshal(meta)
		bulkBuffer.Write(metaJSON)
		bulkBuffer.WriteByte('\n')

		// 添加文档数据行
		docJSON, _ := json.Marshal(doc)
		bulkBuffer.Write(docJSON)
		bulkBuffer.WriteByte('\n')
	}

	// 执行批量插入
	res, err := r.elasticClient.Bulk(
		bytes.NewReader(bulkBuffer.Bytes()),
		r.elasticClient.Bulk.WithIndex(indexName),
	)
	if err != nil {
		return fmt.Errorf("批量插入失败: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		var errorResponse map[string]interface{}
		if err := json.NewDecoder(res.Body).Decode(&errorResponse); err == nil {
			return fmt.Errorf("批量插入错误: %v", errorResponse)
		}
		return fmt.Errorf("批量插入错误: %s", res.String())
	}

	// 解析响应检查错误
	var bulkResponse map[string]interface{}
	if err := json.NewDecoder(res.Body).Decode(&bulkResponse); err != nil {
		return fmt.Errorf("解析批量响应失败: %w", err)
	}

	if bulkResponse["errors"] == true {
		return fmt.Errorf("批量插入存在错误")
	}

	fmt.Printf("✅ 成功插入 %d 个文档到ElasticSearch\n", len(documents))
	return nil
}

// 生成简化向量（4维向量）
func (r *RAGSystem) generateSimpleVector(text string) []float32 {
	vector := make([]float32, 4)
	for i := 0; i < 4; i++ {
		hash := float32(0)
		for j, ch := range text {
			if j >= 10 {
				break
			}
			hash += float32(ch) * float32(i+1)
		}
		vector[i] = hash / 1000.0
	}

	// 归一化
	var norm float32
	for _, v := range vector {
		norm += v * v
	}
	if norm > 0 {
		norm = float32(norm)
		for i := range vector {
			vector[i] /= norm
		}
	}

	return vector
}

// 获取直接答案（纯DeepSeek）
func (r *RAGSystem) GetDirectAnswer(question string) (string, float64, error) {
	start := time.Now()

	ctx := context.Background()
	resp, err := r.openAIClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.config.DeepSeekModel,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "你是一个知识渊博的助手，请基于你的知识回答问题。",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: question,
			},
		},
		Temperature: 0.1,
		MaxTokens:   500,
	})

	if err != nil {
		return "", 0, err
	}

	elapsed := time.Since(start).Seconds()

	if len(resp.Choices) == 0 {
		return "", elapsed, fmt.Errorf("未收到回答")
	}

	return resp.Choices[0].Message.Content, elapsed, nil
}

// 获取RAG增强答案
func (r *RAGSystem) GetRAGAnswer(question string) (string, float64, []SearchResult, error) {
	start := time.Now()

	// 1. 检索相关文档
	results, err := r.SearchDocuments(question, 3)
	if err != nil {
		return "", 0, nil, err
	}

	// 2. 构建上下文
	var contextBuilder strings.Builder
	contextBuilder.WriteString("以下是相关文档信息：\n\n")

	for i, result := range results {
		contextBuilder.WriteString(fmt.Sprintf("文档%d: %s\n", i+1, result.Title))
		contextBuilder.WriteString(fmt.Sprintf("内容: %s\n\n", result.Content))
	}

	ctx := contextBuilder.String()

	// 3. 调用DeepSeek生成答案
	resp, err := r.openAIClient.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: r.config.DeepSeekModel,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "你是一个严谨的AI助手，必须严格基于提供的上下文信息回答问题。如果上下文信息不足，请如实告知。不要编造上下文之外的信息。",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("上下文信息：\n%s\n\n问题：%s\n\n请基于上述上下文信息回答问题：", ctx, question),
			},
		},
		Temperature: 0.1,
		MaxTokens:   500,
	})

	elapsed := time.Since(start).Seconds()

	if err != nil {
		return "", elapsed, results, err
	}

	if len(resp.Choices) == 0 {
		return "", elapsed, results, fmt.Errorf("未收到回答")
	}

	return resp.Choices[0].Message.Content, elapsed, results, nil
}

// 搜索相关文档 - 使用ElasticSearch 8.x 向量搜索
func (r *RAGSystem) SearchDocuments(query string, topK int) ([]SearchResult, error) {
	indexName := r.config.IndexName

	// 生成查询向量
	queryVector := r.generateSimpleVector(query)

	// 方法1：使用ElasticSearch 8.x的script_score进行向量搜索
	// 将float32转换为float64
	vector64 := make([]float64, len(queryVector))
	for i, v := range queryVector {
		vector64[i] = float64(v)
	}

	// 构建搜索查询
	searchQuery := map[string]interface{}{
		"size": topK,
		"query": map[string]interface{}{
			"script_score": map[string]interface{}{
				"query": map[string]interface{}{
					"match_all": map[string]interface{}{},
				},
				"script": map[string]interface{}{
					"source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
					"params": map[string]interface{}{
						"query_vector": vector64,
					},
				},
			},
		},
		"_source": []string{"title", "content"},
	}

	// 执行搜索
	searchJSON, _ := json.Marshal(searchQuery)
	res, err := r.elasticClient.Search(
		r.elasticClient.Search.WithIndex(indexName),
		r.elasticClient.Search.WithBody(bytes.NewReader(searchJSON)),
		r.elasticClient.Search.WithTrackTotalHits(false),
	)
	if err != nil {
		// 如果向量搜索失败，尝试混合搜索
		return r.HybridSearch(query, topK)
	}
	defer res.Body.Close()

	if res.IsError() {
		// 尝试混合搜索作为降级策略
		return r.HybridSearch(query, topK)
	}

	// 解析搜索结果
	var searchResponse map[string]interface{}
	if err := json.NewDecoder(res.Body).Decode(&searchResponse); err != nil {
		return nil, fmt.Errorf("解析搜索结果失败: %w", err)
	}

	var results []SearchResult

	// 检查是否有命中结果
	hits, ok := searchResponse["hits"].(map[string]interface{})
	if !ok {
		return results, nil
	}

	hitsList, ok := hits["hits"].([]interface{})
	if !ok {
		return results, nil
	}

	for _, hit := range hitsList {
		hitMap, ok := hit.(map[string]interface{})
		if !ok {
			continue
		}

		// 获取分数
		score, ok := hitMap["_score"].(float64)
		if !ok {
			score = 0
		}

		// 计算相似度分数（归一化）
		normalizedScore := score / 2.0 // cosineSimilarity返回-1到1，+1后为0-2
		if normalizedScore > 1.0 {
			normalizedScore = 1.0
		}

		// 获取源文档
		source, ok := hitMap["_source"].(map[string]interface{})
		if !ok {
			continue
		}

		// 提取标题和内容
		title, _ := source["title"].(string)
		content, _ := source["content"].(string)

		results = append(results, SearchResult{
			Title:   title,
			Content: content,
			Score:   normalizedScore,
		})

		// 调试输出
		fmt.Printf("找到文档: Title=%s, Score=%.2f\n", title, normalizedScore)
	}

	return results, nil
}

// 混合搜索：向量搜索 + 文本搜索
func (r *RAGSystem) HybridSearch(query string, topK int) ([]SearchResult, error) {
	indexName := r.config.IndexName

	// 方法2：文本搜索（降级策略）
	searchQuery := map[string]interface{}{
		"size": topK,
		"query": map[string]interface{}{
			"multi_match": map[string]interface{}{
				"query":    query,
				"fields":   []string{"title", "content"},
				"type":     "best_fields",
				"operator": "and",
			},
		},
		"_source": []string{"title", "content"},
	}

	searchJSON, _ := json.Marshal(searchQuery)
	res, err := r.elasticClient.Search(
		r.elasticClient.Search.WithIndex(indexName),
		r.elasticClient.Search.WithBody(bytes.NewReader(searchJSON)),
	)
	if err != nil {
		return nil, fmt.Errorf("混合搜索失败: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return nil, fmt.Errorf("混合搜索错误: %s", res.String())
	}

	// 解析搜索结果
	var searchResponse map[string]interface{}
	if err := json.NewDecoder(res.Body).Decode(&searchResponse); err != nil {
		return nil, fmt.Errorf("解析混合搜索结果失败: %w", err)
	}

	var results []SearchResult

	// 检查是否有命中结果
	hits, ok := searchResponse["hits"].(map[string]interface{})
	if !ok {
		return results, nil
	}

	hitsList, ok := hits["hits"].([]interface{})
	if !ok {
		return results, nil
	}

	for _, hit := range hitsList {
		hitMap, ok := hit.(map[string]interface{})
		if !ok {
			continue
		}

		// 获取分数
		score, ok := hitMap["_score"].(float64)
		if !ok {
			score = 0
		}

		// 归一化处理
		normalizedScore := score / 100.0
		if normalizedScore > 1.0 {
			normalizedScore = 1.0
		}

		// 获取源文档
		source, ok := hitMap["_source"].(map[string]interface{})
		if !ok {
			continue
		}

		// 提取标题和内容
		title, _ := source["title"].(string)
		content, _ := source["content"].(string)

		results = append(results, SearchResult{
			Title:   title,
			Content: content,
			Score:   normalizedScore,
		})
	}
	return results, nil
}
