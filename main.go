package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/sashabaranov/go-openai"
)

// 配置结构体
type Config struct {
	MilvusHost     string
	MilvusPort     int
	DeepSeekAPIKey string
	DeepSeekModel  string
	CollectionName string
}

// 文档结构体
type Document struct {
	ID      string
	Title   string
	Content string
	Vector  []float32
}

// 搜索结果
type SearchResult struct {
	Title   string
	Content string
	Score   float32
}

// RAG系统
type RAGSystem struct {
	milvusClient client.Client
	openAIClient *openai.Client
	config       Config
}

func main() {
	fmt.Println("🚀 RAG简易Demo启动...")

	// 加载配置
	config := loadConfig()

	// 创建RAG系统
	rag, err := NewRAGSystem(config)
	if err != nil {
		log.Fatalf("创建RAG系统失败: %v", err)
	}
	defer rag.Close()

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
		fmt.Println("\n🔍 获取纯DeepSeek回答：")
		directAnswer, directTime, err := rag.GetDirectAnswer(question)
		if err != nil {
			fmt.Printf("❌ 获取直接答案失败: %v\n", err)
			continue
		}
		fmt.Printf("⏱️  响应时间: %.2f秒\n", directTime)
		fmt.Printf("💬 回答: %s\n", directAnswer)

		// 获取RAG答案
		fmt.Println("\n🔍 获取RAG增强回答：")
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
	fmt.Println("💡 总结: RAG在需要最新、具体信息的场景表现更好")
	fmt.Println(strings.Repeat("=", 50))
}

// 加载配置
func loadConfig() Config {
	// 加载.env文件
	godotenv.Load()

	return Config{
		MilvusHost:     getEnv("MILVUS_HOST", "localhost"),
		MilvusPort:     getEnvAsInt("MILVUS_PORT", 19530),
		DeepSeekAPIKey: getEnv("DEEPSEEK_API_KEY", ""),
		DeepSeekModel:  getEnv("DEEPSEEK_MODEL", "deepseek-chat"),
		CollectionName: getEnv("COLLECTION_NAME", "rag_demo"),
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
	_, _ = fmt.Sscanf(value, "%d", &result)
	return result
}

// 创建RAG系统
func NewRAGSystem(config Config) (*RAGSystem, error) {
	// 验证配置
	if config.DeepSeekAPIKey == "" {
		return nil, fmt.Errorf("DEEPSEEK_API_KEY不能为空")
	}

	// 连接Milvus
	milvusClient, err := client.NewClient(context.Background(), client.Config{
		Address: fmt.Sprintf("%s:%d", config.MilvusHost, config.MilvusPort),
	})
	if err != nil {
		return nil, fmt.Errorf("连接Milvus失败: %w", err)
	}

	conf := openai.DefaultConfig(config.DeepSeekAPIKey)
	conf.BaseURL = "https://api.deepseek.com"

	return &RAGSystem{
		milvusClient: milvusClient,
		openAIClient: openai.NewClientWithConfig(conf),
		config:       config,
	}, nil
}

// 初始化知识库
func (r *RAGSystem) InitializeKnowledgeBase() error {
	ctx := context.Background()
	collectionName := r.config.CollectionName

	// 检查集合是否存在
	exists, err := r.milvusClient.HasCollection(ctx, collectionName)
	if err != nil {
		return err
	}

	// 如果集合已存在，先删除（为了演示）
	if exists {
		err = r.milvusClient.DropCollection(ctx, collectionName)
		if err != nil {
			return fmt.Errorf("删除集合失败: %w", err)
		}
	}

	// 创建集合
	err = r.milvusClient.CreateCollection(ctx, &entity.Schema{
		CollectionName: collectionName,
		Description:    "RAG演示知识库",
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				AutoID:     false,
				TypeParams: map[string]string{
					"max_length": "100",
				},
			},
			{
				Name:     "title",
				DataType: entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "200",
				},
			},
			{
				Name:     "content",
				DataType: entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "10000",
				},
			},
			{
				Name:     "vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": "4", // 简化版，使用4维向量
				},
			},
		},
		EnableDynamicField: false,
	}, 2) // 分片数为2
	if err != nil {
		return fmt.Errorf("创建集合失败: %w", err)
	}

	// 插入示例文档
	err = r.insertSampleDocuments()
	if err != nil {
		return fmt.Errorf("插入文档失败: %w", err)
	}

	// 创建索引
	index, err := entity.NewIndexHNSW(entity.L2, 8, 64)
	if err != nil {
		return fmt.Errorf("创建索引失败: %w", err)
	}

	err = r.milvusClient.CreateIndex(ctx, collectionName, "vector", index, false)
	if err != nil {
		return fmt.Errorf("创建向量索引失败: %w", err)
	}

	return nil
}

// 插入示例文档
func (r *RAGSystem) insertSampleDocuments() error {
	ctx := context.Background()

	// 示例文档数据（包含最新信息）
	documents := []Document{
		{
			ID:      "doc_001",
			Title:   "闫同学人物介绍",
			Content: "闫同学，男，来自中国，26岁，天蝎座，是知名技术博主、摄影博主、技术爱好者，擅长写Go语言，喜欢打羽毛球。",
		},
		{
			ID:      "doc_002",
			Title:   "扯编程的淡公众号介绍",
			Content: "扯编程的淡，科技领域知名微信公众号，由闫同学运营，内容多为技术博客，日常生活感想，截止2026年1月，已有粉丝2000+。",
		},
	}

	// 为每个文档生成向量并插入
	var ids []string
	var titles []string
	var contents []string
	var vectors [][]float32

	for _, doc := range documents {
		// 生成简化向量（4维）
		vector := r.generateSimpleVector(doc.Content)

		ids = append(ids, doc.ID)
		titles = append(titles, doc.Title)
		contents = append(contents, doc.Content)
		vectors = append(vectors, vector)
	}

	// 插入数据
	idColumn := entity.NewColumnVarChar("id", ids)
	titleColumn := entity.NewColumnVarChar("title", titles)
	contentColumn := entity.NewColumnVarChar("content", contents)
	vectorColumn := entity.NewColumnFloatVector("vector", 4, vectors)

	_, err := r.milvusClient.Insert(ctx, r.config.CollectionName, "", idColumn, titleColumn, contentColumn, vectorColumn)

	if err != nil {
		return err
	}

	fmt.Printf("✅ 插入了 %d 个文档到知识库\n", len(documents))
	return nil
}

// 生成简化向量（4维向量）
func (r *RAGSystem) generateSimpleVector(text string) []float32 {
	// 创建4维向量
	vector := make([]float32, 4)

	// 基于文本内容生成简单的向量表示
	// 这里只是示例，实际应用中应该使用embedding模型
	for i := 0; i < 4; i++ {
		// 简单的哈希函数生成伪随机向量值
		hash := float32(0)
		for j, ch := range text {
			if j >= 10 { // 只取前10个字符
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

	contextStr := contextBuilder.String()

	// 3. 调用DeepSeek生成答案
	ctx := context.Background()
	resp, err := r.openAIClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.config.DeepSeekModel,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "你是一个严谨的AI助手，必须严格基于提供的上下文信息回答问题。如果上下文信息不足，请如实告知。不要编造上下文之外的信息。",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("上下文信息：\n%s\n\n问题：%s\n\n请基于上述上下文信息回答问题：", contextStr, question),
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

// 搜索相关文档 - 使用最新的Milvus SDK API
func (r *RAGSystem) SearchDocuments(query string, topK int) ([]SearchResult, error) {
	ctx := context.Background()
	collectionName := r.config.CollectionName

	// 加载集合
	err := r.milvusClient.LoadCollection(ctx, collectionName, false)
	if err != nil {
		return nil, fmt.Errorf("加载集合失败: %w", err)
	}

	// 生成查询向量
	queryVector := r.generateSimpleVector(query)

	// 搜索参数
	sp, _ := entity.NewIndexHNSWSearchParam(32)

	// 执行搜索 - 根据最新SDK修正
	searchResults, err := r.milvusClient.Search(
		ctx,
		collectionName,
		nil,                          // 分区列表
		"",                           // 表达式
		[]string{"title", "content"}, // 输出字段
		[]entity.Vector{entity.FloatVector(queryVector)}, // 查询向量
		"vector",  // 向量字段名
		entity.L2, // 距离度量
		topK,      // topK
		sp,        // 搜索参数
	)

	if err != nil {
		return nil, fmt.Errorf("搜索失败: %w", err)
	}

	var results []SearchResult

	// 检查是否有结果
	if len(searchResults) == 0 {
		return results, nil
	}

	// 获取第一个查询的结果（因为我们只查询了一个向量）
	if len(searchResults) > 0 {
		searchResult := searchResults[0]

		// 获取ID列
		idCol, ok := searchResult.IDs.(*entity.ColumnVarChar)
		if !ok {
			return results, fmt.Errorf("ID列类型错误")
		}

		// 获取分数列和字段
		scores := searchResult.Scores
		fields := searchResult.Fields

		// 遍历所有结果
		for i := 0; i < searchResult.ResultCount; i++ {
			// 获取ID、分数
			id := idCol.Data()[i]
			score := float64(1.0 / (1.0 + scores[i]))

			// 获取标题和内容
			var title, content string
			for _, field := range fields {
				switch field.Name() {
				case "title":
					if col, ok := field.(*entity.ColumnVarChar); ok {
						title = col.Data()[i]
					}
				case "content":
					if col, ok := field.(*entity.ColumnVarChar); ok {
						content = col.Data()[i]
					}
				}
			}

			// 添加到结果列表
			results = append(results, SearchResult{
				Title:   title,
				Content: content,
				Score:   float32(score),
			})

			// 调试输出
			fmt.Printf("找到文档: ID=%s, Title=%s, Score=%.2f\n", id, title, score)
		}
	}

	return results, nil
}

func (r *RAGSystem) Close() {
	if r.milvusClient != nil {
		if err := r.milvusClient.Close(); err != nil {
			fmt.Println(err)
		}
	}
}
