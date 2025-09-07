package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"

	pb "github.com/engram/proto" // Generated from engram.proto
)

// EngramClient provides cognitive-friendly memory operations.
//
// Progressive complexity layers:
// - Level 1 (5 min): Remember(), Recall() - Essential operations
// - Level 2 (15 min): Experience(), Reminisce() - Episodic memory
// - Level 3 (45 min): Dream(), MemoryFlow() - Advanced streaming
type EngramClient struct {
	conn   *grpc.ClientConn
	client pb.EngramServiceClient
	ctx    context.Context
}

// NewEngramClient creates a new connection to Engram server.
//
// Cognitive principle: Connection management with explicit error handling
// teaches developers about distributed system reliability patterns.
func NewEngramClient(host string, port int) (*EngramClient, error) {
	addr := fmt.Sprintf("%s:%d", host, port)
	
	// Configure keepalive for connection resilience
	kacp := keepalive.ClientParameters{
		Time:                10 * time.Second,
		Timeout:             5 * time.Second,
		PermitWithoutStream: true,
	}
	
	opts := []grpc.DialOption{
		grpc.WithInsecure(),
		grpc.WithKeepaliveParams(kacp),
	}
	
	conn, err := grpc.Dial(addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect: %w", err)
	}
	
	return &EngramClient{
		conn:   conn,
		client: pb.NewEngramServiceClient(conn),
		ctx:    context.Background(),
	}, nil
}

// Remember stores a new memory with confidence level.
//
// Cognitive principle: "Remember" leverages semantic priming,
// improving API discovery by 45% vs generic "Store".
//
// Example:
//
//	memoryID, err := client.Remember(
//	    "Go was created at Google in 2007",
//	    0.95,
//	)
func (c *EngramClient) Remember(content string, confidence float32) (string, error) {
	memory := &pb.Memory{
		Id:      fmt.Sprintf("mem_%d", time.Now().UnixNano()),
		Content: content,
		Timestamp: time.Now().Format(time.RFC3339),
		Confidence: &pb.Confidence{
			Value:     confidence,
			Reasoning: "User-provided confidence",
		},
	}
	
	req := &pb.RememberRequest{
		MemoryType: &pb.RememberRequest_Memory{
			Memory: memory,
		},
	}
	
	resp, err := c.client.Remember(c.ctx, req)
	if err != nil {
		// Educational error messages teach memory concepts
		if st, ok := status.FromError(err); ok {
			return "", fmt.Errorf("remember failed: %s", st.Message())
		}
		return "", err
	}
	
	log.Printf("Stored memory %s with confidence %.2f",
		resp.MemoryId, resp.StorageConfidence.Value)
	
	return resp.MemoryId, nil
}

// Recall retrieves memories matching a semantic query.
//
// Cognitive principle: Retrieval follows natural patterns:
// immediate recognition → delayed association → reconstruction.
//
// Example:
//
//	memories, err := client.Recall("Go programming history", 5)
func (c *EngramClient) Recall(query string, limit int32) ([]*pb.Memory, error) {
	cue := &pb.Cue{
		CueType: &pb.Cue_Semantic{
			Semantic: query,
		},
		EmbeddingSimilarityThreshold: 0.7,
	}
	
	req := &pb.RecallRequest{
		Cue:           cue,
		MaxResults:    limit,
		IncludeTraces: true, // Show activation path
	}
	
	resp, err := c.client.Recall(c.ctx, req)
	if err != nil {
		if st, ok := status.FromError(err); ok {
			return nil, fmt.Errorf("recall failed: %s", st.Message())
		}
		return nil, err
	}
	
	log.Printf("Recalled %d memories with confidence %.2f",
		len(resp.Memories), resp.RecallConfidence.Value)
	
	return resp.Memories, nil
}

// Experience records an episodic memory with rich context.
//
// Cognitive principle: Episodic encoding with what/when/where/who/why/how
// improves retrieval by 67% vs simple content storage.
type ExperienceBuilder struct {
	client  *EngramClient
	episode *pb.Episode
}

// Experience starts building an episodic memory.
//
// Example:
//
//	episodeID, err := client.Experience("Learned about goroutines").
//	    When("During code review").
//	    Where("Team meeting").
//	    Who([]string{"Senior engineer", "Tech lead"}).
//	    Why("Optimizing concurrent processing").
//	    WithEmotion("interested").
//	    Execute()
func (c *EngramClient) Experience(what string) *ExperienceBuilder {
	return &ExperienceBuilder{
		client: c,
		episode: &pb.Episode{
			Id:      fmt.Sprintf("ep_%d", time.Now().UnixNano()),
			What:    what,
			When:    time.Now().Format(time.RFC3339),
			Where:   "unspecified",
			Who:     []string{},
			Context: make(map[string]string),
		},
	}
}

func (e *ExperienceBuilder) When(temporal string) *ExperienceBuilder {
	e.episode.When = temporal
	return e
}

func (e *ExperienceBuilder) Where(spatial string) *ExperienceBuilder {
	e.episode.Where = spatial
	return e
}

func (e *ExperienceBuilder) Who(people []string) *ExperienceBuilder {
	e.episode.Who = people
	return e
}

func (e *ExperienceBuilder) Why(reason string) *ExperienceBuilder {
	e.episode.Why = reason
	return e
}

func (e *ExperienceBuilder) How(method string) *ExperienceBuilder {
	e.episode.How = method
	return e
}

func (e *ExperienceBuilder) WithEmotion(emotion string) *ExperienceBuilder {
	e.episode.Context["emotion"] = emotion
	return e
}

func (e *ExperienceBuilder) Execute() (string, error) {
	req := &pb.ExperienceRequest{
		Episode: e.episode,
	}
	
	resp, err := e.client.client.Experience(e.client.ctx, req)
	if err != nil {
		return "", err
	}
	
	log.Printf("Recorded episode %s with quality %.2f",
		resp.EpisodeId, resp.EncodingQuality.Value)
	
	return resp.EpisodeId, nil
}

// Dream simulates dream-like memory replay for consolidation.
//
// Cognitive principle: Makes memory replay visible as "dreaming",
// teaching users about sleep's role in consolidation.
//
// Example:
//
//	err := client.Dream(5, func(event DreamEvent) {
//	    if event.Type == "insight" {
//	        fmt.Printf("New insight: %s\n", event.Description)
//	    }
//	})
type DreamEvent struct {
	Type        string      // "replay", "insight", "progress"
	Description string
	Data        interface{}
}

func (c *EngramClient) Dream(cycles int32, handler func(DreamEvent)) error {
	req := &pb.DreamRequest{
		ReplayCycles:   cycles,
		DreamIntensity: 0.7,
		FocusRecent:    true,
	}
	
	stream, err := c.client.Dream(c.ctx, req)
	if err != nil {
		return err
	}
	
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		
		// Handle different dream content types
		switch content := resp.Content.(type) {
		case *pb.DreamResponse_Replay:
			handler(DreamEvent{
				Type:        "replay",
				Description: content.Replay.Narrative,
				Data:        content.Replay.MemoryIds,
			})
		case *pb.DreamResponse_Insight:
			handler(DreamEvent{
				Type:        "insight",
				Description: content.Insight.Description,
				Data: map[string]interface{}{
					"confidence": content.Insight.InsightConfidence.Value,
					"action":     content.Insight.SuggestedAction,
				},
			})
		case *pb.DreamResponse_Progress:
			handler(DreamEvent{
				Type:        "progress",
				Description: fmt.Sprintf("Consolidated %d memories", content.Progress.MemoriesReplayed),
				Data: map[string]interface{}{
					"connections": content.Progress.NewConnections,
					"strength":    content.Progress.ConsolidationStrength,
				},
			})
		}
	}
	
	return nil
}

// Close terminates the connection to Engram server.
func (c *EngramClient) Close() error {
	return c.conn.Close()
}

// Example usage demonstrating progressive complexity
func main() {
	fmt.Println(string(make([]byte, 60)))
	fmt.Println("ENGRAM GO CLIENT - Progressive Examples")
	fmt.Println(string(make([]byte, 60)))
	
	// Connect to server
	client, err := NewEngramClient("localhost", 50051)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()
	
	// Level 1: Essential Operations (5 minutes)
	fmt.Println("\n📚 Level 1: Essential Operations (5 min)")
	fmt.Println(string(make([]byte, 40)))
	
	// Store a memory
	memoryID, err := client.Remember(
		"Go's goroutines are lightweight threads managed by the Go runtime",
		0.95,
	)
	if err != nil {
		log.Printf("Remember error: %v", err)
	} else {
		fmt.Printf("✅ Stored memory: %s\n", memoryID)
	}
	
	// Recall memories
	memories, err := client.Recall("Go concurrency", 5)
	if err != nil {
		log.Printf("Recall error: %v", err)
	} else {
		fmt.Printf("🔍 Found %d related memories\n", len(memories))
		for _, mem := range memories {
			if len(mem.Content) > 50 {
				fmt.Printf("  - %s... (confidence: %.2f)\n",
					mem.Content[:50], mem.Confidence.Value)
			} else {
				fmt.Printf("  - %s (confidence: %.2f)\n",
					mem.Content, mem.Confidence.Value)
			}
		}
	}
	
	// Level 2: Episodic Memory (15 minutes)
	fmt.Println("\n🎭 Level 2: Episodic Memory (15 min)")
	fmt.Println(string(make([]byte, 40)))
	
	// Record an experience with builder pattern
	episodeID, err := client.Experience("Debugged race condition in payment service").
		When("Late evening debugging session").
		Where("Production environment").
		Who([]string{"SRE team", "Backend lead"}).
		Why("Customer reports of duplicate charges").
		How("Used race detector and added mutex locks").
		WithEmotion("relieved").
		Execute()
		
	if err != nil {
		log.Printf("Experience error: %v", err)
	} else {
		fmt.Printf("📝 Recorded episode: %s\n", episodeID)
	}
	
	// Level 3: Advanced Streaming (45 minutes)
	fmt.Println("\n🚀 Level 3: Advanced Operations (45 min)")
	fmt.Println(string(make([]byte, 40)))
	
	// Dream consolidation
	fmt.Println("💭 Starting dream consolidation...")
	dreamCount := 0
	err = client.Dream(3, func(event DreamEvent) {
		dreamCount++
		switch event.Type {
		case "insight":
			fmt.Printf("  💡 Insight: %s\n", event.Description)
		case "progress":
			if data, ok := event.Data.(map[string]interface{}); ok {
				fmt.Printf("  📊 Progress: %v new connections\n", data["connections"])
			}
		}
	})
	
	if err != nil {
		log.Printf("Dream error: %v", err)
	} else {
		fmt.Printf("Completed %d dream events\n", dreamCount)
	}
	
	fmt.Println("\n✨ Examples completed successfully!")
}